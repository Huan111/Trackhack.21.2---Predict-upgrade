###########This code is for track hack project of team emotional-suport-vector-machine###########

import matplotlib.pyplot as plt

#plot barplot for different variables
def discrete_var_barplot(x,y,data):
    plt.figure(figsize=(15,10))
    sns.barplot(x=x,y=y,data=data)

#plot boxplot for different variables
def discrete_var_boxplot(x,y,data):
    plt.figure(figsize=(15,10))
    sns.boxplot(x=x,y=y,data=data)
    
#merge 2 different datasets based on different condition
def merge_table(left,right,on,how):
    return pd.merge(left,right,on = on,how = how)

#plot count for different variables
def discrete_var_countplot(x,data):
    plt.figure(figsize=(15,10))
    sns.countplot(x=x,data=data)
   
#Plot the correlation scores in different variables
def correlation_plot(data):
    corrmat = data.corr()
    fig, ax = plt.subplots()
    fig.set_size_inches(15,15)
    sns.heatmap(corrmat,cmap="YlGnBu",linewidths=.5,annot=True)
    
#convert categorical features to numerical representation
def cat_to_num(x,data):
    return data[x].astype('category').cat.codes

#fill na with most frequent value
def fill_na_with_fre(x,data):
    return data[x].fillna(data[x].mode()[0])

#fill na with particular value
def fill_na_with_val(num,na_col,data):
    return data[na_col].fillna(num)

#detect the outlier by IQR method and return outlier numbers and fences
def detect_outlier_IQR(data,col,threshold):
    IQR = data[col].quantile(0.75) - data[col].quantile(0.25)#calculate IQR
    Lower_fence = data[col].quantile(0.25) - (IQR * threshold)#define lower fence
    Upper_fence = data[col].quantile(0.75) + (IQR * threshold)#define higher fence
    tmp = pd.concat([data[col]>Upper_fence,data[col]<Lower_fence],axis=1)#find the outliers mask
    outlier_index = tmp.any(axis=1)
    try:
        outlier_num = outlier_index.value_counts()[1]#outlier numbers
    except:
        outlier_num = 0
    fences = (Lower_fence,Upper_fence)
    return outlier_num, fences

#replace outliers with upper fences or low fences
def windsorization(data,col,fences):
    data_copy = data.copy(deep=True)  
    data_copy.loc[data_copy[col]<fences[0],col] = fences[0]#replace with lower fences
    data_copy.loc[data_copy[col]>fences[1],col] = fences[1]#replace with higher fences
    return data_copy

#relace rare value with different value
def replace_rare_value(data,col,threshold):
    temp_df = pd.Series(data[col].value_counts()/len(data))#calculate count percentages
    mapping = { k: ('other' if k not in temp_df[temp_df >= threshold].index else k)
                for k in temp_df.index}#create rare values mapping
    return data[col].replace(mapping)


from sklearn.feature_selection import chi2

#chi test for feature selections
def chi_test(data,X,y):
    y_val = data[y]#extract target
    chi_scores = chi2(data[X],y_val)#conduct chi test
    p_values = pd.Series(chi_scores[1],index = data[X].columns)#produce p values
    p_values.sort_values(ascending = False , inplace = True)# sort p values
    p_values.plot.bar()#plot
    return p_values

#select the features based on chi test p-values
def select_fea_by_chi(data,p_vals,threshold):
    drop_cols = p_vals[p_vals >= threshold].index#
    return data.drop(drop_cols,axis=1)

import scipy.stats as stats
#student-t test for numerical features
def t_test(temp,X,y):
    population = temp[temp[y] == 0][X].mean()#mean based on target = 0
    return stats.ttest_1samp(a = temp[temp[y]==1][X],popmean = population)# return reuslts of the test

#select features by t test
def select_fea_by_t(data,test,threshold):
    columns = test.statistic.index
    drop_cols = columns[test.pvalue >= threshold]#get the drop columns
    return data.drop(drop_cols,axis=1)
    
import numpy as np
#remove features based on correlation
def remove_features_cor(data,corr_score=0.9):
    corr = data.corr()
    columns = np.full((corr.shape[0],), True, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i+1, corr.shape[0]):
            if corr.iloc[i,j] >= corr_score:#detect the corr score bigger than threshold
                if columns[j]:#set column to false if it is the first
                    columns[j] = False
    select_columns = data.columns[columns]
    return data[select_columns]

#create feature by groupby transform
def groupby_transform(data,col,by,method):
    return data.groupby(by)[col].transform(method)

#create feature by groupby agg
def groupby_agg(data,col,by,func):
    return data[by].map(data.groupby(by)[col].agg(func))

#file paths
teamname = 'emotional-support-vector-machine-unsw'
data_folder='s3://tf-trachack-data/212/'
root_folder='s3://tf-trachack-notebooks/'+teamname+'/jupyter/jovyan/'
import pandas as pd

#create new datasets
def create_new_data(f_type,name):
    data_path = data_folder+"data/" + f_type + '/'
    to_data_path = name + '/new_data_1/' + f_type + '_'
    print('Starting to creating...')
    upgrades=pd.read_csv(data_path + "upgrades.csv")
    
    #customer dataset
    customer_info=pd.read_csv(data_path + "customer_info.csv")
    customer_info['plan_name'] = fill_na_with_fre('plan_name',customer_info)

    from sklearn.preprocessing import RobustScaler
    customer_info['cus_used_days'] = pd.Series(pd.to_datetime(customer_info['redemption_date']) - pd.to_datetime(customer_info['first_activation_date'])).dt.days #days between red and first_activation date
    
    #dealing with outliers
    
    #outlier_num,fences = detect_outlier_IQR(customer_info,'cus_used_days',3)
    #if outlier_num != 0:
    #    customer_info = windsorization(customer_info,'cus_used_days',fences)
    
    scaler = RobustScaler()#scaling
    customer_info['cus_used_days'] = fill_na_with_val(customer_info['cus_used_days'].mean(),'cus_used_days',customer_info)#replace nan by mean
    customer_info['cus_used_days'] = scaler.fit_transform(customer_info['cus_used_days'].values.reshape(-1,1))#robust scaler
    select_features = ['line_id','cus_used_days', 'plan_name','carrier']#final selected features
    #generate new datasets
    new_customer_info = customer_info[select_features].drop_duplicates().reset_index(drop=True)
    new_customer_info.to_csv(root_folder+ to_data_path + "new_customer_info.csv",header=True,index=None)

    #phone_info dataset
    phone_info=pd.read_csv(data_path + "phone_info.csv")
    phone_upg = merge_table(upgrades,phone_info,on='line_id',how='left')
    
    #used features
    temp_feature = ['cpu_cores',
           'expandable_storage', 'gsma_device_type', 'gsma_model_name',
           'gsma_operating_system', 'internal_storage_capacity', 'lte',
           'lte_advanced', 'lte_category', 'manufacturer', 'os_family', 'os_name',
           'os_vendor', 'os_version', 'sim_size', 'total_ram', 'touch_screen',
           'wi_fi', 'year_released']
    
    #label_encoder = LabelEncoder()
    for i in temp_feature:
        phone_upg[i] = fill_na_with_val('ismiss',i,phone_upg)#replace nan with ismiss
        phone_upg[i] = replace_rare_value(phone_upg,i,0.03)#repalce rare value under specific threshold
        #phone_upg[i] = label_encoder.fit_transform(phone_upg[i])
        
    temp = ['expandable_storage','lte','lte_advanced','lte_category','touch_screen','wi_fi','year_released']#categorical features but represent by number
    for i in temp:#transfer them to string type
        phone_upg[i] = phone_upg[i].astype('str')
        
    #conduct chi2 test    
    #chi_test(phone_upg,temp_features,'upgrade')
    #correlation_plot(phone_upg[temp_features])
    #new_phone = remove_features_cor(phone_upg[temp_features],0.88)
    #new_phone = pd.concat((phone_upg['line_id'],remove_features_cor(phone_upg[temp_features],0.88)),axis=1)
    
    #final selected features
    select_features = ['line_id','cpu_cores', 'expandable_storage', 'gsma_device_type',
           'gsma_model_name', 'gsma_operating_system', 'internal_storage_capacity',
           'lte_advanced', 'lte_category', 'os_family', 'os_version', 'sim_size',
           'total_ram', 'year_released']

    #create dataset
    phone_upg[select_features].to_csv(root_folder+ to_data_path + "new_phone_info.csv",header=True,index=None)

    #redemption dataset
    redemptions=pd.read_csv(data_path + "redemptions.csv")
    
    #create new features
    
    redemptions['red_count'] = groupby_transform(redemptions,'channel','line_id','count')#using count in each person as a feature
    redemptions['red_mean_rev'] = groupby_transform(redemptions,'gross_revenue','line_id','mean')#the mean of a person's GR
    redemptions['channel_unique'] = groupby_transform(redemptions,'channel','line_id','nunique')#the number of unique channel the person use
    redemptions['red_type_unique'] = groupby_transform(redemptions,'redemption_type','line_id','nunique')#the numebr of unique red_type 
    redemptions['rev_type_unique'] = groupby_transform(redemptions,'revenue_type','line_id','nunique')#the number of unique revenue_type
    redemptions['channel_most_fre'] = groupby_agg(redemptions,'channel','line_id',lambda x: x.value_counts().idxmax())#the most frequent channel used
    redemptions['red_type_most_fre'] = groupby_agg(redemptions,'redemption_type','line_id',lambda x: x.value_counts().idxmax())#the most frequent red type used
    redemptions['rev_type_most_fre'] = groupby_agg(redemptions,'revenue_type','line_id',lambda x: x.value_counts().idxmax())#the most revenue_type used
    
    #select used features
    use_feature = ['line_id','red_count','red_mean_rev','channel_unique','red_type_unique','rev_type_unique','channel_most_fre','red_type_most_fre','rev_type_most_fre']
    
    #drop duplicates rows and merge with upgrades
    new_redemptions = redemptions[use_feature].drop_duplicates().reset_index(drop=True)
    new_redemptions = merge_table(upgrades,new_redemptions,'line_id','left')

    #fill nan withh most frequent values
    for i in use_feature[1:]:
        new_redemptions[i] = fill_na_with_fre(i,new_redemptions)#fill nan with most frequent

    #dealring with outlier 
    for i in ['red_count','red_mean_rev']:
        outlier_num,fences = detect_outlier_IQR(new_redemptions,i,3)#detect outlier by IQR
        if outlier_num != 0:
            new_redemptions = windsorization(new_redemptions,i,fences)#replace outliers by windsorizetion method
    
    #replace rare values
    new_redemptions['channel_most_fre'] = replace_rare_value(new_redemptions,'channel_most_fre',0.02)
    new_redemptions['red_type_most_fre'] = replace_rare_value(new_redemptions,'red_type_most_fre',0.02)
    new_redemptions['rev_type_most_fre'] = replace_rare_value(new_redemptions,'rev_type_most_fre',0.02)
    
    #label encode
    #label_encoder = LabelEncoder()
    #for i in cat_features:
    #    new_redemptions[i] = label_encoder.fit_transform(new_redemptions[i])

    #conduct t-test and chi2-test
    #p_vals = chi_test(new_redemptions,['channel_most_fre','red_type_most_fre','rev_type_most_fre','channel_unique','red_type_unique','rev_type_unique'],'upgrade')
    #new_redemptions = select_fea_by_chi(new_redemptions,p_vals,0.05)
    #temp = t_test(new_redemptions,['red_count','red_mean_rev'],'upgrade')
    #new_redemptions = select_fea_by_t(new_redemptions,temp,0.05)
    
    #scaling
    scaler = RobustScaler()
    for i in ['red_count','red_mean_rev']:
        new_redemptions[i] = scaler.fit_transform(new_redemptions[i].values.reshape(-1,1))
        
    #final selected features
    select_features = ['line_id','red_count', 'red_mean_rev',
           'channel_unique', 'red_type_unique', 'rev_type_unique',
           'channel_most_fre', 'red_type_most_fre', 'rev_type_most_fre']
    #new dataset
    new_redemptions[select_features].to_csv(root_folder+ to_data_path + "new_redemptions.csv",header=True,index=None)

    #deactivations and reactivations datasets merge
    deactivations=pd.read_csv(data_path + "deactivations.csv")
    reactivations=pd.read_csv(data_path + "reactivations.csv")
    
    #merge datasets
    dea_rea_info = merge_table(deactivations,reactivations,on='line_id',how='inner')
    dea_rea_upg = merge_table(upgrades,dea_rea_info,'line_id','left')
    
    #create new features
    dea_rea_upg['deactivation_reason'] = fill_na_with_fre('deactivation_reason',dea_rea_upg)#fill nan with most frequent
    dea_rea_upg['reactivation_channel'] = fill_na_with_fre('reactivation_channel',dea_rea_upg)
    dea_rea_upg['de_re_counts'] = groupby_transform(dea_rea_upg,'deactivation_date','line_id','count')#use count as new feature
    dea_rea_upg['reason_unique'] = groupby_transform(dea_rea_upg,'deactivation_reason','line_id','nunique')#unique count values
    dea_rea_upg['de_re_channel_unique'] = groupby_transform(dea_rea_upg,'reactivation_channel','line_id','nunique')#unique count values
    dea_rea_upg['de_re_channel_most_fre'] = groupby_agg(dea_rea_upg,'reactivation_channel','line_id',lambda x: x.value_counts().idxmax())#most frequent values
    dea_rea_upg['de_re_reason_most_fre'] = groupby_agg(dea_rea_upg,'deactivation_reason','line_id',lambda x: x.value_counts().idxmax())#most frequent values
    
    #used features and frop duplicate rows
    use_features = ['line_id','de_re_counts','reason_unique','de_re_channel_unique','de_re_channel_most_fre','de_re_reason_most_fre']
    new_dea_rea = dea_rea_upg[use_features].drop_duplicates().reset_index(drop=True)
    
    #replace rare values
    for i in ['de_re_channel_most_fre','de_re_reason_most_fre']:
        new_dea_rea[i] = replace_rare_value(new_dea_rea,i,0.02)
    
    #outlier dealing
    outlier_num,fences = detect_outlier_IQR(new_dea_rea,'de_re_counts',3)
    new_dea_rea[outlier_index]
    if outlier_num!= 0:
        new_dea_rea = windsorization(new_dea_rea,'de_re_counts',fences)
    
    #conduct chi2 test and t test
    #cat_features = ['de_re_channel_most_fre','de_re_reason_most_fre']
    #for i in cat_features:
    #    new_dea_rea[i] = label_encoder.fit_transform(new_dea_rea[i])
    #new_dea_rea['upgrade'] = dea_rea_upg['upgrade']
    #p_vals = chi_test(new_dea_rea,cat_features,'upgrade')
    #new_dea_rea = select_fea_by_chi(new_dea_rea,p_vals,0.05)
    #use_features = ['de_re_counts','reason_unique','channel_unique']
    #temp = t_test(new_dea_rea,use_features,'upgrade')
    #new_dea_rea = select_fea_by_t(new_dea_rea,temp,0.05)
    
    #final selecting features
    select_features = ['line_id', 'de_re_counts', 'reason_unique', 'de_re_channel_unique',
           'de_re_channel_most_fre', 'de_re_reason_most_fre']
    #new datasets
    new_dea_rea[select_features].to_csv(root_folder+ to_data_path + "new_rea_dea.csv",header=True,index=None)

    #suspensions dataset
    suspensions=pd.read_csv(data_path + "suspensions.csv")
    
    #new fatures
    suspensions['sus_count'] = groupby_transform(suspensions,'suspension_start_date','line_id','count')#using count as new features
    suspensions = suspensions[['line_id','sus_count']].drop_duplicates().reset_index(drop=True)#drop duplicates rows
    new_suspensions = merge_table(upgrades,suspensions,'line_id','left')#merge with upgrade
    
    new_suspensions['sus_count'] = fill_na_with_val(0,'sus_count',new_suspensions)#replace nan with 0 sus count
    
    #scaler
    scaler = RobustScaler()
    new_suspensions['sus_count'] = scaler.fit_transform(new_suspensions['sus_count'].values.reshape(-1,1))
    #new dataset
    new_suspensions[['line_id','sus_count']].to_csv(root_folder+ to_data_path + "new_suspensions.csv",header=True,index=None)

    #newwork dataset
    network_usage_domestic=pd.read_csv(data_path + "network_usage_domestic.csv")
    
    #new featrues
    
    network_usage_domestic['net_work_mean_kb'] = groupby_transform(network_usage_domestic,'total_kb','line_id','mean')#total_kb mean

    network_usage_domestic['net_mms_in_mean'] = groupby_transform(network_usage_domestic,'mms_in','line_id','mean')#mms_in mean
    network_usage_domestic['net_mms_out_mean'] = groupby_transform(network_usage_domestic,'mms_out','line_id','mean')#mms_out mean
    network_usage_domestic['net_mms_mean_sum'] = network_usage_domestic['net_mms_in_mean'] + network_usage_domestic['net_mms_out_mean']#mms mean sum

    network_usage_domestic['net_sms_in_mean'] = groupby_transform(network_usage_domestic,'sms_in','line_id','mean')#sms_in mean
    network_usage_domestic['net_sms_out_mean'] = groupby_transform(network_usage_domestic,'sms_out','line_id','mean')#sms_out mean
    network_usage_domestic['net_sms_mean_sum'] = network_usage_domestic['net_sms_in_mean'] + network_usage_domestic['net_sms_out_mean']#sms mean sum

    network_usage_domestic['net_voice_count_in_mean'] = groupby_transform(network_usage_domestic,'voice_count_in','line_id','mean')#voice count in mean
    network_usage_domestic['voice_count_out'] = network_usage_domestic['voice_count_total'] - network_usage_domestic['voice_count_in']#voice count out
    network_usage_domestic['net_voice_count_out_mean'] = groupby_transform(network_usage_domestic,'voice_count_out','line_id','mean')#voice count out mean
    network_usage_domestic['net_voice_count_mean_sum'] = network_usage_domestic['net_voice_count_in_mean'] + network_usage_domestic['net_voice_count_out_mean']#voice count mean sum

    network_usage_domestic['net_voice_min_in_mean'] = groupby_transform(network_usage_domestic,'voice_min_in','line_id','mean')#voice minute in mean
    network_usage_domestic['net_voice_min_out_mean'] = groupby_transform(network_usage_domestic,'voice_min_out','line_id','mean')#voice minute out mean
    network_usage_domestic['net_voice_min_mean_sum'] = network_usage_domestic['net_voice_min_in_mean'] + network_usage_domestic['net_voice_min_out_mean']#voice minute mean sum

    network_usage_domestic['net_mms_ratio'] = network_usage_domestic['net_mms_in_mean'] / network_usage_domestic['net_mms_out_mean']#mms ratio
    network_usage_domestic['net_sms_ratio'] = network_usage_domestic['net_sms_in_mean'] / network_usage_domestic['net_sms_out_mean']#sms ratio
    network_usage_domestic['net_voice_min_ratio'] = network_usage_domestic['net_voice_min_in_mean'] / network_usage_domestic['net_voice_min_out_mean']#voice minute ratio
    network_usage_domestic['net_voice_count_ratio'] = network_usage_domestic['net_voice_count_in_mean'] / network_usage_domestic['net_voice_count_out_mean']#voice count ratio

    network_usage_domestic['net_work_count'] = groupby_transform(network_usage_domestic,'date','line_id','count')#count number

    #final used features
    select_features = ['net_work_mean_kb','net_mms_mean_sum','net_sms_mean_sum','net_voice_count_mean_sum','net_voice_min_mean_sum',
                       'net_mms_ratio','net_sms_ratio','net_voice_min_ratio','net_voice_count_ratio','net_work_count']
    
    #drop duplicated features andmerge datasets
    network_usage_domestic = network_usage_domestic[['line_id'] + select_features]
    network_usage_domestic = network_usage_domestic.drop_duplicates().reset_index(drop=True)
    new_network_usage_domestic = merge_table(upgrades,network_usage_domestic,'line_id','left')
    
    #scaling and dealing with outliers
    scaler = RobustScaler()
    for i in select_features:
        new_network_usage_domestic[i] = fill_na_with_val(new_network_usage_domestic[i].mean(),i,new_network_usage_domestic)#fill with mean
        #deal with outliers
        outlier_num,fences = detect_outlier_IQR(new_network_usage_domestic,i,8)
        if outlier_num !=0:
            new_network_usage_domestic = windsorization(new_network_usage_domestic,i,fences)
        new_network_usage_domestic[i] = scaler.fit_transform(new_network_usage_domestic[i].values.reshape(-1,1))#scaling
    
    #new datasets
    new_network_usage_domestic[['line_id'] + select_features].to_csv(root_folder+ to_data_path + "new_network_usage_domestic.csv",header=True,index=None)
    #end of feature engineering
    print('Finished')
    
import pandas as pd
#merger all datasets
def merge_tables(f_type,name):
    """
    merge the tables, f_type: 'dev' or 'eval'
    must create new_data folder under your working folder
    """
    print('Start to merge...')
    data_path = name + '/new_data_1/' + f_type + '_'#file path
    
    #dataset to merge
    new_redemptions = pd.read_csv(root_folder+ data_path + "new_redemptions.csv")
    new_phone_info = pd.read_csv(root_folder+ data_path + "new_phone_info.csv")
    new_customer_info = pd.read_csv(root_folder+ data_path +"new_customer_info.csv")
    new_deactivation = pd.read_csv(root_folder+data_path + "new_rea_dea.csv")
    new_suspension = pd.read_csv(root_folder+ data_path +"new_suspensions.csv")
    new_network_usage_domestic = pd.read_csv(root_folder+ data_path +"new_network_usage_domestic.csv")
    upgrades=pd.read_csv(data_folder+"data/" + f_type + "/upgrades.csv")

    table_list = [new_redemptions,new_phone_info,new_customer_info,new_suspension,new_network_usage_domestic,new_deactivation,upgrades]
    final_merge = pd.concat(table_list, join='inner', axis=1)#inner join
    final_merge = final_merge.loc[:,~final_merge.columns.duplicated()]#drop duplicated columns
    final_merge.to_csv(root_folder + data_path + "final_merge_ver4.csv",header=True,index=None)#create new datasets
    print('Finished')

#run above functions
name = 'guohuan-li'
merge_tables('dev',name)
merge_tables('eval',name)

data_path = name + '/new_data_1/' + 'dev' + '_'
merge_train = pd.read_csv(root_folder+ data_path + "final_merge_ver4.csv")

data_path = name + '/new_data_1/' + 'eval' + '_'
merge_val = pd.read_csv(root_folder+ data_path + "final_merge_ver4.csv")

#replace values that not matched
merge_train['os_version'].replace('6.0.1','other',inplace=True)

#define categorical features and numerical features
cat_features = ['channel_most_fre',
       'red_type_most_fre', 'rev_type_most_fre', 'cpu_cores',
       'expandable_storage', 'gsma_device_type', 'gsma_model_name',
       'gsma_operating_system', 'internal_storage_capacity', 'lte_advanced',
       'lte_category', 'os_family', 'os_version', 'sim_size', 'total_ram',
       'year_released','plan_name', 'carrier','de_re_channel_most_fre',
       'de_re_reason_most_fre']
num_featrues = [ 'red_count', 'red_mean_rev', 'channel_unique',
       'red_type_unique', 'rev_type_unique','cus_used_days','sus_count',
       'net_work_mean_kb', 'net_mms_mean_sum', 'net_sms_mean_sum',
       'net_voice_count_mean_sum', 'net_voice_min_mean_sum', 'net_mms_ratio',
       'net_sms_ratio', 'net_voice_min_ratio', 'net_voice_count_ratio',
       'net_work_count', 'de_re_counts']

#ordinal encoding for categorical features
from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder()
for i in cat_features:
    encoder.fit(merge_train[i].values.reshape(-1,1))
    merge_train[i] = encoder.transform(merge_train[i].values.reshape(-1,1))#tranform in trainng datasets
    merge_val[i] = encoder.transform(merge_val[i].values.reshape(-1,1))#transform in validation datastes
    
#define file path
data_path = name + '/new_data_1/' + 'eval' + '_'
merge_val.to_csv(root_folder + data_path + "final_merge_ver4_ord.csv",header=True,index=None)

data_path = name' + '/new_data_1/' + 'dev' + '_'
merge_train.to_csv(root_folder + data_path + "final_mergever4_ord.csv",header=True,index=None)