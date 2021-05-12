####ver2
####The order of feature engineering
####1. Data explore: Barplot,boxplot, see any outlier and categorical features number
####2. Missing value fill: numerical fill with mean or mode, categorical fill with 'ismiss' or mode
####
import matplotlib.pyplot as plt
def discrete_var_barplot(x,y,data):
    plt.figure(figsize=(15,10))
    sns.barplot(x=x,y=y,data=data)

def discrete_var_boxplot(x,y,data):
    plt.figure(figsize=(15,10))
    sns.boxplot(x=x,y=y,data=data)
    
def merge_table(left,right,on,how):
    return pd.merge(left,right,on = on,how = how)

def discrete_var_countplot(x,data):
    plt.figure(figsize=(15,10))
    sns.countplot(x=x,data=data)
   
#The correlation map
def correlation_plot(data):
    corrmat = data.corr()
    fig, ax = plt.subplots()
    fig.set_size_inches(15,15)
    sns.heatmap(corrmat,cmap="YlGnBu",linewidths=.5,annot=True)
    
#convert categorical features to number
def cat_to_num(x,data):
    return data[x].astype('category').cat.codes

#fill na with most frequent value
def fill_na_with_fre(x,data):
    return data[x].fillna(data[x].mode()[0])

#fill na with perticular value
def fill_na_with_val(num,na_col,data):
    return data[na_col].fillna(num)

#detect the outlier by IQR
def detect_outlier_IQR(data,col,threshold):
    IQR = data[col].quantile(0.75) - data[col].quantile(0.25)
    Lower_fence = data[col].quantile(0.25) - (IQR * threshold)
    Upper_fence = data[col].quantile(0.75) + (IQR * threshold)
    tmp = pd.concat([data[col]>Upper_fence,data[col]<Lower_fence],axis=1)
    outlier_index = tmp.any(axis=1)
    try:
        outlier_num = outlier_index.value_counts()[1]
    except:
        outlier_num = 0
    return outlier_index, outlier_num

#replace outlier by mean or most frequent values
def replace_outlier(data,col,idx,method):
    data_copy = data.copy(deep=True)
    if method == 'mean':
        data_copy.loc[idx,col] = data_copy[col].mean()
    else:
        data_copy.loc[idx,col] = data_copy[col].mode()[0]
    return data_copy

#relace rare value with different value
def replace_rare_value(data,col,threshold):
    temp_df = pd.Series(data[col].value_counts()/len(data))
    mapping = { k: ('other' if k not in temp_df[temp_df >= threshold].index else k)
                for k in temp_df.index}
    return data[col].replace(mapping)


from sklearn.feature_selection import chi2

#chi test for feature selections
def chi_test(data,X,y):
    y_val = data[y]
    chi_scores = chi2(data[X],y_val)
    p_values = pd.Series(chi_scores[1],index = data[X].columns)
    p_values.sort_values(ascending = False , inplace = True)
    p_values.plot.bar()
    return p_values

#select the features based on chi test p-values
def select_fea_by_chi(data,p_vals,threshold):
    drop_cols = p_vals[p_vals >= threshold].index
    return data.drop(drop_cols,axis=1)

import scipy.stats as stats
#student-t test for numerical features
def t_test(temp,X,y):
    population = temp[temp[y] == 0][X].mean()
    return stats.ttest_1samp(a = temp[temp[y]==1][X],popmean = population)

#select features by t test
def select_fea_by_t(data,test,threshold):
    columns = test.statistic.index
    drop_cols = columns[test.pvalue >= threshold]
    return data.drop(drop_cols,axis=1)
    
import numpy as np
#remove features based on correlation
def remove_features_cor(data,corr_score=0.9):
    corr = data.corr()
    columns = np.full((corr.shape[0],), True, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i+1, corr.shape[0]):
            if corr.iloc[i,j] >= corr_score:
                if columns[j]:
                    columns[j] = False
    select_columns = data.columns[columns]
    return data[select_columns]
#create feature by groupby transform
def groupby_transform(data,col,by,method):
    return data.groupby(by)[col].transform(method)

#create feature by groupby agg
def groupby_agg(data,col,by,func):
    return data[by].map(data.groupby(by)[col].agg(func))

data_path = data_folder+"data/" + f_type + '/'
to_data_path = name + '/new_data_1/' + f_type + '_'
print('Starting to creating...')
upgrades=pd.read_csv(data_path + "upgrades.csv")
upgrades['upgrade'].replace({'yes':1,'no':0},inplace = True)

customer_info=pd.read_csv(data_path + "customer_info.csv")
customer_info['plan_name'] = fill_na_with_fre('plan_name',customer_info)
#customer_info['plan_name'] = cat_to_num('plan_name',customer_info)
#customer_info['carrier'] = cat_to_num('carrier',customer_info)

from sklearn.preprocessing import RobustScaler
customer_info['cus_used_days'] = pd.Series(pd.to_datetime(customer_info['redemption_date']) - pd.to_datetime(customer_info['first_activation_date'])).dt.days
outlier_idx,outlier_num = detect_outlier_IQR(customer_info,'cus_used_days',3)
if outlier_num != 0:
    customer_info = replace_outlier(customer_info,'cus_used_days',outlier_idx,'mean')
scaler = RobustScaler()
customer_info['cus_used_days'] = fill_na_with_val(-999,'cus_used_days',customer_info)
customer_info['cus_used_days'] = scaler.fit_transform(customer_info['cus_used_days'].values.reshape(-1,1))
select_features = ['line_id','cus_used_days', 'plan_name','carrier']
new_customer_info = customer_info[select_features].drop_duplicates().reset_index(drop=True)
new_customer_info.to_csv(root_folder+ to_data_path + "new_customer_info.csv",header=True,index=None)

phone_info=pd.read_csv(data_path + "phone_info.csv")
phone_upg = merge_table(upgrades,phone_info,on='line_id',how='left')
temp = ['expandable_storage','lte','lte_advanced','lte_category','touch_screen','wi_fi','year_released']

temp_feature = ['cpu_cores',
       'expandable_storage', 'gsma_device_type', 'gsma_model_name',
       'gsma_operating_system', 'internal_storage_capacity', 'lte',
       'lte_advanced', 'lte_category', 'manufacturer', 'os_family', 'os_name',
       'os_vendor', 'os_version', 'sim_size', 'total_ram', 'touch_screen',
       'wi_fi', 'year_released']
#label_encoder = LabelEncoder()
for i in temp_feature:
    phone_upg[i] = fill_na_with_val('ismiss',i,phone_upg)
    phone_upg[i] = replace_rare_value(phone_upg,i,0.03)
    #phone_upg[i] = label_encoder.fit_transform(phone_upg[i])
for i in temp:
    phone_upg[i] = phone_upg[i].astype('str')
#chi_test(phone_upg,temp_features,'upgrade')
#correlation_plot(phone_upg[temp_features])
#new_phone = remove_features_cor(phone_upg[temp_features],0.88)
#new_phone = pd.concat((phone_upg['line_id'],remove_features_cor(phone_upg[temp_features],0.88)),axis=1)
select_features = ['line_id','cpu_cores', 'expandable_storage', 'gsma_device_type',
       'gsma_model_name', 'gsma_operating_system', 'internal_storage_capacity',
       'lte_advanced', 'lte_category', 'os_family', 'os_version', 'sim_size',
       'total_ram', 'year_released']

phone_upg[select_features].to_csv(root_folder+ to_data_path + "new_phone_info.csv",header=True,index=None)

redemptions=pd.read_csv(data_path + "redemptions.csv")
redemptions['red_count'] = groupby_transform(redemptions,'channel','line_id','count')
redemptions['red_mean_rev'] = groupby_transform(redemptions,'gross_revenue','line_id','mean')
redemptions['channel_unique'] = groupby_transform(redemptions,'channel','line_id','nunique')
redemptions['red_type_unique'] = groupby_transform(redemptions,'redemption_type','line_id','nunique')
redemptions['rev_type_unique'] = groupby_transform(redemptions,'revenue_type','line_id','nunique')
redemptions['channel_most_fre'] = groupby_agg(redemptions,'channel','line_id',lambda x: x.value_counts().idxmax())
redemptions['red_type_most_fre'] = groupby_agg(redemptions,'redemption_type','line_id',lambda x: x.value_counts().idxmax())
redemptions['rev_type_most_fre'] = groupby_agg(redemptions,'revenue_type','line_id',lambda x: x.value_counts().idxmax())
use_feature = ['line_id','red_count','red_mean_rev','channel_unique','red_type_unique','rev_type_unique','channel_most_fre','red_type_most_fre','rev_type_most_fre']
new_redemptions = redemptions[use_feature].drop_duplicates().reset_index(drop=True)
new_redemptions = merge_table(upgrades,new_redemptions,'line_id','left')

for i in use_feature[1:]:
    new_redemptions[i] = fill_na_with_fre(i,new_redemptions)

for i in ['red_count','red_mean_rev']:
    outlier_idx,outlier_num = detect_outlier_IQR(new_redemptions,i,3)
    if outlier_num != 0:
        new_redemptions = replace_outlier(new_redemptions,i,outlier_idx,'mean')

new_redemptions['channel_most_fre'] = replace_rare_value(new_redemptions,'channel_most_fre',0.02)
new_redemptions['red_type_most_fre'] = replace_rare_value(new_redemptions,'red_type_most_fre',0.02)
new_redemptions['rev_type_most_fre'] = replace_rare_value(new_redemptions,'rev_type_most_fre',0.02)
cat_features = ['channel_most_fre','red_type_most_fre','rev_type_most_fre']
#label_encoder = LabelEncoder()
#for i in cat_features:
#    new_redemptions[i] = label_encoder.fit_transform(new_redemptions[i])

#p_vals = chi_test(new_redemptions,['channel_most_fre','red_type_most_fre','rev_type_most_fre','channel_unique','red_type_unique','rev_type_unique'],'upgrade')
#new_redemptions = select_fea_by_chi(new_redemptions,p_vals,0.05)
#temp = t_test(new_redemptions,['red_count','red_mean_rev'],'upgrade')
#new_redemptions = select_fea_by_t(new_redemptions,temp,0.05)
scaler = RobustScaler()
for i in ['red_count','red_mean_rev']:
    new_redemptions[i] = scaler.fit_transform(new_redemptions[i].values.reshape(-1,1))
select_features = ['line_id',  'red_count', 'red_mean_rev','channel_most_fre', 'red_type_most_fre', 'rev_type_most_fre']
new_redemptions[select_features].to_csv(root_folder+ to_data_path + "new_redemptions.csv",header=True,index=None)

suspensions=pd.read_csv(data_path + "suspensions.csv")
suspensions['sus_count'] = groupby_transform(suspensions,'suspension_start_date','line_id','count')
suspensions = suspensions[['line_id','sus_count']].drop_duplicates().reset_index(drop=True)
new_suspensions = merge_table(upgrades,suspensions,'line_id','left')
new_suspensions['sus_count'] = fill_na_with_fre('sus_count',new_suspensions)
outlier_index,outlier_num = detect_outlier_IQR(new_suspensions,'sus_count',3)
if outlier_num!=0:
    new_suspensions = replace_outlier(new_suspensions,'sus_count',outlier_index,'mean')
scaler = RobustScaler()
new_suspensions['sus_count'] = scaler.fit_transform(new_suspensions['sus_count'].values.reshape(-1,1))
new_suspensions[['line_id','sus_count']].to_csv(root_folder+ to_data_path + "new_suspensions.csv",header=True,index=None)

network_usage_domestic=pd.read_csv(data_path + "network_usage_domestic.csv")
network_usage_domestic['net_work_mean_kb'] = groupby_transform(network_usage_domestic,'total_kb','line_id','mean')
network_usage_domestic['net_work_count'] = groupby_transform(network_usage_domestic,'date','line_id','count')
network_usage_domestic = network_usage_domestic[['line_id','net_work_mean_kb','net_work_count']]
network_usage_domestic = network_usage_domestic.drop_duplicates().reset_index(drop=True)
new_network_usage_domestic = merge_table(upgrades,network_usage_domestic,'line_id','left')
scaler = RobustScaler()
for i in ['net_work_mean_kb','net_work_count']:
    new_network_usage_domestic[i] = fill_na_with_val(new_network_usage_domestic[i].mean(),i,new_network_usage_domestic)
    outlier_index,outlier_num = detect_outlier_IQR(new_network_usage_domestic,i,3)
    if outlier_num !=0:
        new_network_usage_domestic = replace_outlier(new_network_usage_domestic,i,outlier_index,'mean')
    new_network_usage_domestic[i] = scaler.fit_transform(new_network_usage_domestic[i].values.reshape(-1,1))
new_network_usage_domestic[['line_id','net_work_mean_kb','net_work_count']].to_csv(root_folder+ to_data_path + "new_network_usage_domestic.csv",header=True,index=None)

def merge_tables(f_type,name):
    """
    merge the tables, f_type: 'dev' or 'eval'
    must create new_data folder under your working folder
    """
    print('Start to merge...')
    data_path = name + '/new_data_1/' + f_type + '_'

    new_redemptions = pd.read_csv(root_folder+ data_path + "new_redemptions.csv")
    new_phone_info = pd.read_csv(root_folder+ data_path + "new_phone_info.csv")
    new_customer_info = pd.read_csv(root_folder+ data_path +"new_customer_info.csv")
    #new_deactivation = pd.read_csv(root_folder+data_path + "new_rea_dea.csv")
    new_suspension = pd.read_csv(root_folder+ data_path +"new_suspensions.csv")
    new_network_usage_domestic = pd.read_csv(root_folder+ data_path +"new_network_usage_domestic.csv")
    upgrades=pd.read_csv(data_folder+"data/" + f_type + "/upgrades.csv")

    table_list = [new_redemptions,new_phone_info,new_customer_info,new_suspension,new_network_usage_domestic,upgrades]
    final_merge = pd.concat(table_list, join='inner', axis=1)
    final_merge = final_merge.loc[:,~final_merge.columns.duplicated()]
    final_merge.to_csv(root_folder + data_path + "final_merge.csv",header=True,index=None)
    print('Finished')
    
merge_tables('dev',name)
merge_tables('eval',name)

data_path = name + '/new_data_1/' + 'dev' + '_'
merge_train = pd.read_csv(root_folder+ data_path + "final_merge.csv")

data_path = name + '/new_data_1/' + 'eval' + '_'
merge_val = pd.read_csv(root_folder+ data_path + "final_merge.csv")

merge_train['os_version'].replace('6.0.1','other',inplace=True)

cat_features = ['channel_most_fre',
       'red_type_most_fre', 'rev_type_most_fre', 'cpu_cores',
        'gsma_device_type', 'gsma_model_name',
       'gsma_operating_system', 'internal_storage_capacity', 'lte_advanced',
       'lte_category', 'os_family', 'os_version', 'sim_size', 'total_ram',
       'year_released',  'plan_name', 'carrier']

from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder()
for i in cat_features:
    encoder.fit(merge_train[i].values.reshape(-1,1))
    merge_train[i]=encoder.transform(merge_train[i].values.reshape(-1,1))
    merge_val[i] = encoder.transform(merge_val[i].values.reshape(-1,1))
    
data_path = name + '/new_data/' + 'eval' + '_'
merge_val.to_csv(root_folder + data_path + "final_merge.csv",header=True,index=None)

data_path = name + '/new_data/' + 'dev' + '_'
merge_train.to_csv(root_folder + data_path + "final_merge.csv",header=True,index=None)