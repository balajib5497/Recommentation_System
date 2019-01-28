import numpy as np
import pandas as pd

import math

from sklearn.preprocessing import MinMaxScaler
from dateparser import parse
import calendar

def load_data():
    prop        = pd.read_csv('../Dataset/Property dataset/Properties.csv')
    acc_prop    = pd.read_csv('../Dataset/training dataset/Accounts_properties.csv')
    acc         = pd.read_csv('../Dataset/training dataset/Accounts.csv')
    deal_prop   = pd.read_csv('../Dataset/training dataset/Deals_to_Properties.csv')
    opp         = pd.read_csv('../Dataset/training dataset/Opportunities.csv')
    test_data   = pd.read_csv('../Dataset/Test dataset/Test_Data.csv')
    return prop, acc_prop, acc, deal_prop, opp, test_data

def split_date(date, scaled=False):
    if type(date)==float and math.isnan(date):
        return np.nan, np.nan, np.nan
    dt = parse(date)
    
    if scaled:
        num_days = calendar.monthrange(dt.year, dt.month)[1]
        return (dt.day-1)/(num_days-1), (dt.month-1)/11, dt.year
    return dt.day, dt.month, dt.year

def get_scaled_data(data, feature_range=(0,1)):
    scaler = MinMaxScaler(feature_range=feature_range)
    scaled_data = scaler.fit_transform(data)
    scaled_data = pd.DataFrame(scaled_data, columns = data.columns)
    return scaled_data

def clean_acc(acc):
    acc.drop(columns=['consultant', 'master_servicer', 'loan_servicing'], axis=1, inplace=True)
    bool_cols = acc.select_dtypes(include=['bool']).columns
    
    acc[bool_cols] = acc[bool_cols].replace({True:1, False:0})
    acc = pd.get_dummies(acc, columns=['investor_type'], drop_first=False)
    
#    acc['non_actives'] = acc['activity_count'] - acc['active_deals']
#    acc['number_of_failed_deals_as_client'] = acc['num_deals_as_client'] - acc['number_of_won_deals_as_client']
    
    acc['active_deals'] = acc['active_deals'].apply(np.log1p)
    acc['activity_count'] = acc['activity_count'].apply(lambda x: x**(1/3))
    acc['num_deals_as_client'] = acc['num_deals_as_client'].apply(lambda x: x**(2/3))
    acc['num_deals_as_investor'] = acc['num_deals_as_investor'].apply(lambda x: x**(1/3))
    acc['number_of_related_deals'] = acc['number_of_related_deals'].apply(np.log1p)
    acc['number_of_won_deals_as_client'] = acc['number_of_won_deals_as_client'].apply(lambda x: x**(2/3))
    acc['number_of_properties'] = acc['number_of_properties'].apply(np.log1p)
    acc['number_of_related_properties'] = acc['number_of_related_properties'].apply(np.log1p)
    
    col_to_scale = ['active_deals', 'activity_count','non_actives',
                    'num_deals_as_investor', 'number_of_related_deals', 'number_of_won_deals_as_client',
                    'number_of_properties', 'number_of_related_properties'
                    'num_deals_as_client', 'number_of_failed_deals_as_client',
                    ]
    
    acc[col_to_scale] = get_scaled_data(acc[col_to_scale])
    return acc


def clean_opp(opp):
    opp.drop(columns = ['id_deals.1', 'fiscal', 'fiscalyear', 'debt_yield', 'property_type'],
         axis=1, inplace=True)

    opp_closedate = opp['closedate'].apply(split_date, args=(True,))
    opp['closedate_day'] = [dt[0] for dt in opp_closedate]
    opp['closedate_month'] = [dt[1] for dt in opp_closedate]
    opp['closedate_year'] = [dt[2] for dt in opp_closedate]
    del opp_closedate
    
    #opp_date_closed = opp['date_closed'].apply(split_date, args=(True,))
    #opp['date_closed_day'] = [dt[0] for dt in opp_date_closed]
    #opp['date_closed_month'] = [dt[1] for dt in opp_date_closed]
    #opp['date_closed_year'] = [dt[2] for dt in opp_date_closed]
    #del opp_date_closed
    
    #opp_accounting_date = opp['accounting_date'].apply(split_date, args=(True,))
    #opp['accounting_date_day'] = [dt[0] for dt in opp_accounting_date]
    #opp['accounting_date_month'] = [dt[1] for dt in opp_accounting_date]
    #opp['accounting_date_year'] = [dt[2] for dt in opp_accounting_date]
    #del opp_accounting_date
    
    opp.drop(columns = ['closedate', 'date_closed', 'accounting_date'],
             axis=1, inplace=True)
    
    cols_to_normalize = ['closedate_year']
    opp[cols_to_normalize] = get_scaled_data(opp[cols_to_normalize])
    
    opp = pd.get_dummies(opp, columns=['fiscalquarter', 'deal_type', 'platform', 'property_group'], drop_first=False)
    opp['best_initial_bid'].fillna(value=0, inplace=True)
    return opp
    
def clean_prop(prop):
    prop_null = prop.isnull().sum(axis=1)
    prop = prop.loc[prop_null[prop_null < 10].index, :]
    prop.reset_index(drop=True, inplace=True)
    
    prop.drop(columns = ['id_deals', 'building_tax_expenses',
                         'num_parking_spaces', 'price_per_unit', 'size_units',
                         'sale_date__c', 'year_built'],
             axis=1, inplace=True)
    
    prop['market'] = prop['market'].apply(lambda x:x if type(x)==float else int(x[7:]))
    prop['city'] = prop['city'].apply(lambda x:x if type(x)==float else int(x[5:]))
    prop['county'] = prop['county'].apply(lambda x:x if type(x)==float else int(x[7:]))
    
    prop['building_status'].fillna(value='NotDefined', inplace = True)
    prop['property_type_1'].fillna(value='NotDefined', inplace = True)
    
    
    means1 = prop.groupby(by=['county', 'building_status', 'property_type_1']).agg({
                             'num_buildings':np.nanmean,
                             'num_floors':np.nanmean,
                             'occupancy_rate':np.nanmean,
                             'price_per_sq_ft':np.nanmean,
                             'sale_amount__c':np.nanmean,
                             'size_acres':np.nanmean,
                             'size_sf':np.nanmean,
                        })
    means1.reset_index(inplace=True)
    
    cols_to_fill = ['num_buildings', 'num_floors', 'occupancy_rate',
                    'price_per_sq_ft', 'sale_amount__c', 'size_acres',
                    'size_sf']
    for i in prop.index:
        prop_f = means1[(means1['county'] == prop.loc[i,'county']) &
                      (means1['building_status'] == prop.loc[i,'building_status']) &
                      (means1['property_type_1'] == prop.loc[i,'property_type_1']) 	
                      ]
        if prop_f.shape[0] == 0:
            continue
        for col in cols_to_fill:
            if math.isnan(prop.loc[i,col]):
                prop.loc[i,col] = prop_f[col].values[0]
                
                
    means2 = prop.groupby(by=['building_status', 'property_type_1']).agg({
                             'num_buildings':np.nanmean,
                             'num_floors':np.nanmean,
                             'occupancy_rate':np.nanmean,
                             'price_per_sq_ft':np.nanmean,
                             'sale_amount__c':np.nanmean,
                             'size_acres':np.nanmean,
                             'size_sf':np.nanmean,
                        })
    means2.reset_index(inplace=True)
    
    for i in prop.index:
        prop_f = means2[
                      (means2['building_status'] == prop.loc[i,'building_status']) &
                      (means2['property_type_1'] == prop.loc[i,'property_type_1']) 	
                      ]
        if prop_f.shape[0] == 0:
            continue
        for col in cols_to_fill:
            if math.isnan(prop.loc[i,col]):
                prop.loc[i,col] = prop_f[col].values[0]
              
    prop = pd.get_dummies(prop, columns=['class', 'sale_status', 'region__c'], drop_first=False)
    
    prop.loc[prop[prop['market'].isnull()].index,'market'] = 258
    
    prop.loc[prop[prop['occupancy_rate'].isnull()].index, 'occupancy_rate']  \
                = np.nanmean(prop[prop['property_type_1']=='Other']['occupancy_rate'])
                
    prop.loc[prop[prop['sale_amount__c'].isnull()].index, 'sale_amount__c']  \
                = np.nanmean(prop[prop['property_type_1']=='Other']['sale_amount__c'])
                
    prop = pd.get_dummies(prop, columns=['building_status', 'property_type_1'], drop_first=False)
    prop['portfolio'].replace({True:1, False:0}, inplace=True)
    
    prop['num_buildings'] = prop['num_buildings'].apply(np.log1p)
    prop['num_floors'] = prop['num_floors'].apply(np.log1p)
    prop['occupancy_rate'] = prop['occupancy_rate'].apply(lambda x: x**(2/3))
    prop['price_per_sq_ft'] = prop['price_per_sq_ft'].apply(np.log1p)
    prop['sale_amount__c'] = prop['sale_amount__c'].apply(np.log1p)
    prop['size_acres'] = prop['size_acres'].apply(abs).apply(np.log1p)
    prop['size_sf'] = prop['size_sf'].apply(abs).apply(np.log1p)
    
    cols_to_scale = ['num_buildings', 'num_floors', 'occupancy_rate', 'price_per_sq_ft', 
                     'sale_amount__c', 'size_acres', 'size_sf']
    
    prop[cols_to_scale] = get_scaled_data(prop[cols_to_scale])
    
    prop = pd.get_dummies(prop, columns=['county', 'city', 'market'], drop_first=False)
    return prop

def prepare_deals(deal_prop, opp):
    deals = pd.merge(deal_prop, opp, how='inner', on='id_deals')
    deals = deals.groupby('id_props').agg({
                'id_deals':len,
                'best_initial_bid':np.mean, 
                'deal_update_flag':sum,
                'deal_type_Accruing Loan':sum, 
                'deal_type_Advisory':sum, 
                'deal_type_B-Note':sum,
                'deal_type_Bond Enhancement':sum,
                'deal_type_Construction Loan':sum,
                'deal_type_Construction Mini-Perm':sum, 
                'deal_type_Consulting':sum,
                'deal_type_Consulting Fees-Debt':sum, 
                'deal_type_Discretionary Funds':sum,
                'deal_type_Entity Level':sum, 
                'deal_type_Entity Level-Equity':sum,
                'deal_type_Fixed-rate':sum, 
                'deal_type_Floating-Rate':sum,
                'deal_type_Foreclosed/REO':sum, 
                'deal_type_GP Equity':sum,
                'deal_type_Higher Risk Performing':sum, 
                'deal_type_Joint Venture Equity':sum,
                'deal_type_Mezzanine':sum,
                'deal_type_Miscellaneous':sum,
                'deal_type_Non-Performing':sum,
                'deal_type_Participating/Convertible':sum,
                'deal_type_Preferred Equity':sum,
                'deal_type_Property Sale':sum,
                'deal_type_Sale Leaseback':sum,
                'deal_type_Sub-Performing':sum,
                'deal_type_Subscription Facility':sum, 
                'deal_type_Well-Performing':sum,
                'platform_Debt':sum,
                'platform_Equity Placement':sum,
                'platform_Investment Sales':sum, 
                'platform_Loan Sales':sum,
                'platform_Securities':sum,
                'property_group_Healthcare':sum,
                'property_group_Hotel-Lodging':sum, 
                'property_group_Industrial':sum,
                'property_group_Land':sum,
                'property_group_Mixed Use':sum,
                'property_group_Multi-Housing':sum, 
                'property_group_Office':sum,
                'property_group_Other':sum,
                'property_group_Retail':sum,
                'property_group_Self Storage':sum,
                'fiscalquarter_1':sum,
                'fiscalquarter_2':sum,
                'fiscalquarter_3':sum,
                'fiscalquarter_4':sum,
                })
    deals.reset_index(drop=False, inplace=True)
    cols_to_scale = list(deals.columns)
    cols_to_scale.remove('id_props')
    deals[cols_to_scale] = get_scaled_data(deals[cols_to_scale])
    return deals