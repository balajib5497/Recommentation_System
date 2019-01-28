import numpy as np
import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt

#from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
import util

print('loading data...')
prop, acc_prop, acc, deal_prop, opp, test_data = util.load_data()

print('processing data...')
acc = util.clean_acc(acc)
opp = util.clean_opp(opp)
deals = util.prepare_deals(deal_prop, opp)
prop = util.clean_prop(prop)

prop = pd.merge(prop, pd.DataFrame(deals['id_props']), how='inner', on='id_props')
test_data = pd.merge(test_data, acc, how='left', on='id_accs')

print('building interaction matrix...')
dict_id_props = {}
for id_accs in acc.id_accs:
    if id_accs not in list(test_data.id_accs):
        dict_id_props[id_accs] = list(acc_prop[acc_prop['id_accs'] == id_accs]['id_props'])

id_accs_index = { value:key for key, value in acc.id_accs.to_dict().items()}
id_props_index = { value:key for key, value in prop.id_props.to_dict().items()}

alpha = 10
init = 1
interaction_matrix = np.empty((acc.shape[0], prop.shape[0]), dtype=np.int8)
interaction_matrix[:,:] = init
    
for id_accs in dict_id_props.keys():
        for id_props in dict_id_props[id_accs]:
            if id_props in id_props_index:
                interaction_matrix[id_accs_index[id_accs], id_props_index[id_props]] += alpha
                
print('computing user-to-user similarity...')
acc_p = acc.drop('id_accs', axis=1)
acc_corr_all = acc_p.T.corr()

print('finding popular properties...')
prop_freq = deal_prop.groupby(by='id_props').agg({'id':len})
prop_freq.reset_index(drop=False, inplace=True)
prop_freq.columns = ['id_props', 'freq']

prop_freq = pd.merge(prop_freq, pd.DataFrame(prop['id_props']), how='inner', on='id_props')
prop_freq.sort_values(by='freq', ascending=False, inplace=True)

print('running recommendation for test ids...')
test_index = [i for i in acc.index if acc.loc[i, 'id_accs'] in list(test_data['id_accs'])]
acc_corr_all = acc_corr_all[test_index]

n = 10
acc_corr_1 = {}
for col in acc_corr_all.columns:
    acc_corr_1[col] = list(acc_corr_all[col].sort_values(ascending=False)[1:n+1].index)
   
print('filtering properties based on user similarity...')
s = 0.53
acc_corr_neg_1 = {}
for col in acc_corr_all.columns:
   acc_corr_neg_1[col] = list(acc_corr_all[acc_corr_all[col]  < s].index)
   
neg_id_accs = set()
for val in acc_corr_neg_1.values():
    neg_id_accs.update(val)
    
index_to_drop = []
for i in neg_id_accs:
    for j in range(interaction_matrix.shape[1]):
        if interaction_matrix[i,j] > init:
            index_to_drop.append(j)
            
index_to_drop = set(index_to_drop)
prop.drop(index=index_to_drop, axis=0, inplace=True)
prop.reset_index(drop=True, inplace=True)

prop_p = prop.drop('id_props', axis=1)

print('computing item-to-item similarity...')
#prop_corr_all_l1 = pd.DataFrame(euclidean_distances(prop_p, prop_p))
prop_corr_all_l1 = prop_p.T.corr()

prop_corr_1 = {}
for col in prop_corr_all_l1.columns:
    prop_corr_1[col] = list(prop_corr_all_l1[col].sort_values(ascending=True).index)

def get_similar_prop(indices, n_prop=2):
    props = set(indices)
    for i in indices:
        props.update(list(prop_corr_all_l1[col].sort_values(ascending=True)[1:n_prop+2].index))
    return list(props)

print('getting properties(items) of neighbours...')
id_accs_index = { value:key for key, value in acc.id_accs.to_dict().items()}
id_props_index = { value:key for key, value in prop.id_props.to_dict().items()}


alpha = 10
init = 1
interaction_matrix = np.empty((acc.shape[0], prop.shape[0]), dtype=np.int8)
interaction_matrix[:,:] = init
    
for id_accs in dict_id_props.keys():
        for id_props in dict_id_props[id_accs]:
            if id_props in id_props_index:
                interaction_matrix[id_accs_index[id_accs], id_props_index[id_props]] += alpha

recoms_sim = {}
recoms_sim_dict = {}
for id_accs, id_accs_list in acc_corr_1.items():
    recoms_sim[id_accs] = []
    recoms_sim_dict[id_accs] = {}
    for k in range(interaction_matrix.shape[1]):
        count, suma = 0, 0
        for val in interaction_matrix[id_accs_list, k]:
            if val > init:
                count += 1
                suma += val
        if count > 0:
            interaction_matrix[id_accs, k] = round(suma/count)
            if k not in recoms_sim_dict[id_accs]:
                recoms_sim_dict[id_accs][k] = True
                recoms_sim[id_accs].append(k)
    
recom_k = 6
popular_k = 2

for i in acc_corr_1.keys():
    recoms_sim[i] = recoms_sim[i][:recom_k]
    
for i in acc_corr_1.keys():
    recoms_sim[i] = recoms_sim[i]
    
print("getting properties(items) similar to neighbour's properties...")
#for i in acc_corr_1.keys():
#     recoms_sim[i] = get_similar_prop(recoms_sim[i], n_prop=1)

print('storing results...')
results_sim = pd.DataFrame(columns=['id_accs', 'id_prop'])

for i in acc_corr_1.keys():
    top_k = prop.loc[recoms_sim[i], 'id_props'].values.tolist()
    if len(top_k) == 0:
        top_k += prop_freq['id_props'].head(popular_k).values.tolist()
        top_k = list(set(top_k))
    k = len(top_k)
    t = pd.concat([pd.Series([acc.loc[i, 'id_accs']] * k), pd.Series(top_k)], axis=1)
    t.columns = ['id_accs', 'id_prop']
    results_sim = pd.concat([results_sim, t], axis=0, ignore_index=True)
    
results_sim.to_csv('../results/submission_sim.csv', index=False)
counts = results_sim.groupby(by = 'id_accs').agg({'id_prop':len})
counts.reset_index(drop=False, inplace=True)




