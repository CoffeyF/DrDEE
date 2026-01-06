import torch
import torch.nn as nn
from schema import get_event_role_mapping, get_base_event_type, IDX_TO_EVENT_TYPE

torch.set_printoptions(profile="full")

class EventRoleExtractor(nn.Module):
    def __init__(self, config, word_type_vocab_size, event_type_vocab, role_vocab):
        super(EventRoleExtractor, self).__init__()
        self.config = config
        num_embeddings, embed_dim = config.token_embedding.shape
        self.token_embedding = nn.Embedding(num_embeddings, embed_dim)
        self.token_embedding.weight = nn.Parameter(config.token_embedding, requires_grad=False)
        self.word_type_embedding = nn.Embedding(word_type_vocab_size, config.word_type_embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.event_type_embedding = nn.Embedding(len(event_type_vocab['itos']), config.event_type_embedding_dim)
        self.role_embedding = nn.Embedding(len(role_vocab['itos']), config.role_embedding_dim)
        self.event_type_name_to_id = event_type_vocab['stoi']
        self.role_name_to_id = role_vocab['stoi']
        input_dim = config.word_embedding_dim + config.word_type_embedding_dim
        self.token_bilstm = nn.LSTM(input_size=input_dim, hidden_size=config.hidden_size, bidirectional=True, batch_first=True, num_layers=config.num_layers)
        self.event_type_lstm = nn.LSTM(input_size=config.event_type_embedding_dim, hidden_size=config.event_type_embedding_dim, bidirectional=True, batch_first=True, num_layers=1)
        self.gate_layer = nn.Sequential(nn.Linear(config.event_type_embedding_dim + config.role_embedding_dim, config.role_embedding_dim), nn.Sigmoid())
        self.fusion_mlp = nn.Sequential(nn.Linear(config.event_type_embedding_dim + config.role_embedding_dim, config.role_embedding_dim), nn.ReLU(), nn.Linear(config.role_embedding_dim, config.role_embedding_dim))
        self.layer_norm = nn.LayerNorm(config.role_embedding_dim)
        concat_dim = 2 * config.hidden_size + config.role_embedding_dim
        layers = [nn.Linear(concat_dim, config.final_hidden_size), nn.LeakyReLU()]
        for _ in range(config.num_mlps - 1):
            layers += [nn.Linear(config.final_hidden_size, config.final_hidden_size), nn.LeakyReLU()]
        self.mlp_layers = nn.Sequential(*layers)
        self.classifier = nn.Linear(config.final_hidden_size, 2)

    def forward(self, word_ids, wType_ids, event_role_ids):
        token_features = self.token_embedding(word_ids)
        token_features = self.dropout(token_features)
        word_type_features = self.word_type_embedding(wType_ids)
        word_type_features = self.dropout(word_type_features)
        combined_token_features = torch.cat([token_features, word_type_features], dim=1)
        token_lstm_output, _ = self.token_bilstm(combined_token_features.unsqueeze(0))
        token_lstm_output = self.dropout(token_lstm_output).squeeze(0)
        _, idx_to_event_role = get_event_role_mapping()
        device = token_lstm_output.device
        event_type_ids_list = []
        role_ids_list = []
        for eid in event_role_ids:
            if eid.item() == 0:
                event_type_ids_list.append(0)
                role_ids_list.append(0)
            else:
                event_type, role = idx_to_event_role[eid.item()]
                event_type_ids_list.append(self.event_type_name_to_id[event_type])
                base_event_type = get_base_event_type(event_type)
                role_ids_list.append(self.role_name_to_id[(base_event_type, role)])
        event_type_ids_tensor = torch.tensor(event_type_ids_list, dtype=torch.long, device=device)
        role_ids_tensor = torch.tensor(role_ids_list, dtype=torch.long, device=device)
        event_embeddings = self.event_type_embedding(event_type_ids_tensor)
        enhanced_event_embeddings = self._extract_event_type_features(event_embeddings, event_type_ids_tensor)
        role_embeddings = self.role_embedding(role_ids_tensor)
        concatenated = torch.cat([enhanced_event_embeddings, role_embeddings], dim=-1)
        gate_values = self.gate_layer(concatenated)
        mlp_output = self.fusion_mlp(concatenated)
        fused_embeddings = gate_values * role_embeddings + (1 - gate_values) * enhanced_event_embeddings
        fused_embeddings = fused_embeddings + mlp_output
        fused_embeddings = self.layer_norm(fused_embeddings)
        num_words = token_lstm_output.size(0)
        num_event_roles = fused_embeddings.size(0)
        token_expanded = token_lstm_output.unsqueeze(1).expand(-1, num_event_roles, -1)
        event_role_expanded = fused_embeddings.unsqueeze(0).expand(num_words, -1, -1)
        final_features = torch.cat([token_expanded, event_role_expanded], dim=2)
        final_features = final_features.reshape(-1, final_features.size(-1))
        mlp_output = self.mlp_layers(final_features)
        logits = self.classifier(mlp_output)
        return logits

    def _extract_event_type_features(self, event_embeddings, event_type_ids):
        enhanced_embeddings = torch.zeros_like(event_embeddings)
        base_type_to_instance_ids = {}
        for idx, event_type in IDX_TO_EVENT_TYPE.items():
            base_type = get_base_event_type(event_type)
            if base_type not in base_type_to_instance_ids:
                base_type_to_instance_ids[base_type] = []
            base_type_to_instance_ids[base_type].append(idx)
        for base_type, instance_ids in base_type_to_instance_ids.items():
            mask = torch.zeros_like(event_type_ids, dtype=torch.bool)
            for instance_id in instance_ids:
                mask |= (event_type_ids == instance_id)
            if mask.sum() <= 1:
                enhanced_embeddings[mask] = event_embeddings[mask]
                continue
            same_base_type_embeddings = event_embeddings[mask]
            lstm_output, _ = self.event_type_lstm(same_base_type_embeddings.unsqueeze(0))
            lstm_output = lstm_output.squeeze(0)
            enhanced_same_type = lstm_output[:, :event_embeddings.size(-1)] + lstm_output[:, event_embeddings.size(-1):]
            enhanced_embeddings[mask] = enhanced_same_type
        return enhanced_embeddings

EDEE = EventRoleExtractor
