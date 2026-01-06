EVENT_ROLES = {
    'EquityFreeze1': ['EquityHolder', 'FrozeShares', 'LegalInstitution', 'TotalHoldingShares', 'TotalHoldingRatio', 'StartDate', 'EndDate', 'UnfrozeDate'],
    'EquityFreeze2': ['EquityHolder', 'FrozeShares', 'LegalInstitution', 'TotalHoldingShares', 'TotalHoldingRatio', 'StartDate', 'EndDate', 'UnfrozeDate'],
    'EquityFreeze3': ['EquityHolder', 'FrozeShares', 'LegalInstitution', 'TotalHoldingShares', 'TotalHoldingRatio', 'StartDate', 'EndDate', 'UnfrozeDate'],
    'EquityFreeze4': ['EquityHolder', 'FrozeShares', 'LegalInstitution', 'TotalHoldingShares', 'TotalHoldingRatio', 'StartDate', 'EndDate', 'UnfrozeDate'],
    'EquityFreeze5': ['EquityHolder', 'FrozeShares', 'LegalInstitution', 'TotalHoldingShares', 'TotalHoldingRatio', 'StartDate', 'EndDate', 'UnfrozeDate'],
    'EquityFreeze6': ['EquityHolder', 'FrozeShares', 'LegalInstitution', 'TotalHoldingShares', 'TotalHoldingRatio', 'StartDate', 'EndDate', 'UnfrozeDate'],
    'EquityRepurchase1': ['CompanyName', 'HighestTradingPrice', 'LowestTradingPrice', 'RepurchasedShares', 'ClosingDate', 'RepurchaseAmount'],
    'EquityRepurchase2': ['CompanyName', 'HighestTradingPrice', 'LowestTradingPrice', 'RepurchasedShares', 'ClosingDate', 'RepurchaseAmount'],
    'EquityRepurchase3': ['CompanyName', 'HighestTradingPrice', 'LowestTradingPrice', 'RepurchasedShares', 'ClosingDate', 'RepurchaseAmount'],
    'EquityRepurchase4': ['CompanyName', 'HighestTradingPrice', 'LowestTradingPrice', 'RepurchasedShares', 'ClosingDate', 'RepurchaseAmount'],
    'EquityRepurchase5': ['CompanyName', 'HighestTradingPrice', 'LowestTradingPrice', 'RepurchasedShares', 'ClosingDate', 'RepurchaseAmount'],
    'EquityRepurchase6': ['CompanyName', 'HighestTradingPrice', 'LowestTradingPrice', 'RepurchasedShares', 'ClosingDate', 'RepurchaseAmount'],
    'EquityUnderweight1': ['EquityHolder', 'TradedShares', 'StartDate', 'EndDate', 'LaterHoldingShares', 'AveragePrice'],
    'EquityUnderweight2': ['EquityHolder', 'TradedShares', 'StartDate', 'EndDate', 'LaterHoldingShares', 'AveragePrice'],
    'EquityUnderweight3': ['EquityHolder', 'TradedShares', 'StartDate', 'EndDate', 'LaterHoldingShares', 'AveragePrice'],
    'EquityUnderweight4': ['EquityHolder', 'TradedShares', 'StartDate', 'EndDate', 'LaterHoldingShares', 'AveragePrice'],
    'EquityUnderweight5': ['EquityHolder', 'TradedShares', 'StartDate', 'EndDate', 'LaterHoldingShares', 'AveragePrice'],
    'EquityUnderweight6': ['EquityHolder', 'TradedShares', 'StartDate', 'EndDate', 'LaterHoldingShares', 'AveragePrice'],
    'EquityOverweight1': ['EquityHolder', 'TradedShares', 'StartDate', 'EndDate', 'LaterHoldingShares', 'AveragePrice'],
    'EquityOverweight2': ['EquityHolder', 'TradedShares', 'StartDate', 'EndDate', 'LaterHoldingShares', 'AveragePrice'],
    'EquityOverweight3': ['EquityHolder', 'TradedShares', 'StartDate', 'EndDate', 'LaterHoldingShares', 'AveragePrice'],
    'EquityOverweight4': ['EquityHolder', 'TradedShares', 'StartDate', 'EndDate', 'LaterHoldingShares', 'AveragePrice'],
    'EquityOverweight5': ['EquityHolder', 'TradedShares', 'StartDate', 'EndDate', 'LaterHoldingShares', 'AveragePrice'],
    'EquityOverweight6': ['EquityHolder', 'TradedShares', 'StartDate', 'EndDate', 'LaterHoldingShares', 'AveragePrice'],
    'EquityPledge1': ['Pledger', 'PledgedShares', 'Pledgee', 'TotalHoldingShares', 'TotalHoldingRatio', 'TotalPledgedShares', 'StartDate', 'EndDate', 'ReleasedDate'],
    'EquityPledge2': ['Pledger', 'PledgedShares', 'Pledgee', 'TotalHoldingShares', 'TotalHoldingRatio', 'TotalPledgedShares', 'StartDate', 'EndDate', 'ReleasedDate'],
    'EquityPledge3': ['Pledger', 'PledgedShares', 'Pledgee', 'TotalHoldingShares', 'TotalHoldingRatio', 'TotalPledgedShares', 'StartDate', 'EndDate', 'ReleasedDate'],
    'EquityPledge4': ['Pledger', 'PledgedShares', 'Pledgee', 'TotalHoldingShares', 'TotalHoldingRatio', 'TotalPledgedShares', 'StartDate', 'EndDate', 'ReleasedDate'],
    'EquityPledge5': ['Pledger', 'PledgedShares', 'Pledgee', 'TotalHoldingShares', 'TotalHoldingRatio', 'TotalPledgedShares', 'StartDate', 'EndDate', 'ReleasedDate'],
    'EquityPledge6': ['Pledger', 'PledgedShares', 'Pledgee', 'TotalHoldingShares', 'TotalHoldingRatio', 'TotalPledgedShares', 'StartDate', 'EndDate', 'ReleasedDate']
}

EVENT_TYPE_TO_IDX = {'EquityFreeze1':0,'EquityFreeze2':1,'EquityFreeze3':2,'EquityFreeze4':3,'EquityFreeze5':4,'EquityFreeze6':5,'EquityRepurchase1':6,'EquityRepurchase2':7,'EquityRepurchase3':8,'EquityRepurchase4':9,'EquityRepurchase5':10,'EquityRepurchase6':11,'EquityUnderweight1':12,'EquityUnderweight2':13,'EquityUnderweight3':14,'EquityUnderweight4':15,'EquityUnderweight5':16,'EquityUnderweight6':17,'EquityOverweight1':18,'EquityOverweight2':19,'EquityOverweight3':20,'EquityOverweight4':21,'EquityOverweight5':22,'EquityOverweight6':23,'EquityPledge1':24,'EquityPledge2':25,'EquityPledge3':26,'EquityPledge4':27,'EquityPledge5':28,'EquityPledge6':29}

IDX_TO_EVENT_TYPE = {0:'EquityFreeze1',1:'EquityFreeze2',2:'EquityFreeze3',3:'EquityFreeze4',4:'EquityFreeze5',5:'EquityFreeze6',6:'EquityRepurchase1',7:'EquityRepurchase2',8:'EquityRepurchase3',9:'EquityRepurchase4',10:'EquityRepurchase5',11:'EquityRepurchase6',12:'EquityUnderweight1',13:'EquityUnderweight2',14:'EquityUnderweight3',15:'EquityUnderweight4',16:'EquityUnderweight5',17:'EquityUnderweight6',18:'EquityOverweight1',19:'EquityOverweight2',20:'EquityOverweight3',21:'EquityOverweight4',22:'EquityOverweight5',23:'EquityOverweight6',24:'EquityPledge1',25:'EquityPledge2',26:'EquityPledge3',27:'EquityPledge4',28:'EquityPledge5',29:'EquityPledge6'}

def get_event_role_mapping():
    all_pairs = []
    for event_type, roles in EVENT_ROLES.items():
        for role in roles:
            all_pairs.append((event_type, role))
    event_role_to_idx = {'non_conn': 0}
    for idx, pair in enumerate(all_pairs):
        event_role_to_idx[pair] = idx + 1
    idx_to_event_role = {idx: pair for pair, idx in event_role_to_idx.items()}
    return event_role_to_idx, idx_to_event_role

def get_base_event_type(event_type):
    if event_type.endswith('1') or event_type.endswith('2') or event_type.endswith('3') or event_type.endswith('4') or event_type.endswith('5') or event_type.endswith('6'):
        return event_type[:-1]
    return event_type

