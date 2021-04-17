############this module is for feature engineering##############
############ make sure you have new_data folder create under your working folder#####
###### usage: main(t_type,name)

import pandas as pd
teamname = 'emotional-support-vector-machine-unsw'
data_folder='s3://tf-trachack-data/212/'
root_folder='s3://tf-trachack-notebooks/'+teamname+'/jupyter/jovyan/'

def create_new_data(f_type,name):
    '''
        Create new data set, should have created new_data folder in your own directory,
        f_type: 'dev' or 'eval'
    '''
    print('Start to creating ....')
    data_path = data_folder+"data/" + f_type + '/'
    to_data_path = name + '/new_data/' + f_type + '_'
    
    upgrades=pd.read_csv(data_path + "upgrades.csv")
    
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()

    #customer_info
    customer_info=pd.read_csv(data_path + "customer_info.csv")
    customer_info['cus_used_days'] = pd.Series(pd.to_datetime(customer_info['redemption_date']) - pd.to_datetime(customer_info['first_activation_date'])).dt.days
    customer_info['cus_used_days'] = scaler.fit_transform(customer_info['cus_used_days'].values.reshape(-1,1))
    customer_info['cus_used_days'].fillna(-1,inplace = True)
    customer_info['plan_name'].fillna(customer_info['plan_name'].mode()[0], inplace=True)

    new_customer_info = pd.get_dummies(customer_info,columns=['carrier', 'plan_name'],drop_first=True)
    select_features = ['line_id','cus_used_days', 'carrier_carrier 2', 'carrier_carrier 3','plan_name_plan 1', 'plan_name_plan 2', 'plan_name_plan 3','plan_name_plan 4']
    new_customer_info = new_customer_info[select_features].drop_duplicates().reset_index(drop=True)
    new_customer_info.to_csv(root_folder+ to_data_path + "new_customer_info.csv",header=True,index=None)

    #phone_info
    phone_info=pd.read_csv(data_path + "phone_info.csv")
    phone_info['display_description'].fillna('is_miss',inplace=True)
    type_list = ['HUAWEI H226C',  'Motorola XT2052DL', 'ZTE Z899VL',
            'SAMSUNG S327VL', 'IPHONE 6', 'SAMSUNG G950U',
           'IPHONE XR', 'SAMSUNG S260DL', 'LG L52VL', 'LG L322DL',
           'SAMSUNG S727VL', 'IPHONE 7 PLUS', 'SAMSUNG S767VL',
           'IPHONE 6S PLUS', 'HUAWEI H258C', 'Samsung S102DL',
            'LG L555DL', 'IPHONE 7', 
            'IPHONE 5S', 'LM L413DL', 'ALCATEL A502DL',
           'LG L58VL', 'Samsung S215DL', 'SAMSUNG S367VL', 'SAMSUNG S320VL',
           'Samsung S111DL', 'KonnectOne K779HSDL', 'Motorola XT2005DL',
           'iPhone SE 2', 'LG L455DL', 'SAMSUNG G970U1', 'ALCATEL A405DL',
           'SAMSUNG S903VL', 'LG 441G', 'LG LML212VL', 'KonnectOne K500HPEL',
           'MOTOROLA XT1925DL', 'LG L355DL', 'HUAWEI H710VL',
           'SAMSUNG G930VL', 'LG L423DL', 'MOTOROLA XT1955DL', 'ZTE Z723EL',
           'IPHONE SE', 'IPHONE 8', 'IPHONE 8 PLUS', 'SAMSUNG G960U1',
           'LM L713DL', 'BLU B110DL', 'LG L164VL', 'ALCATEL A503DL',
           'SAMSUNG S120VL', 'SAMSUNG S820L', 'LG LM414DL', 'LG L84VL',
           'Samsung S515DL', 'Samsung S115DL', 'Samsung S205DL', 'LG L63BL',
           'ZTE Z917VL', 'Motorola XT2041DL', 'LG L82VL', 'ZTE Z291DL',
           'ZTE Z799VL', 'SAMSUNG G965U1', 'SAMSUNG S907VL', 'SAMSUNG S357BL',
           'LG L158VL', 'LG L125DL', 'IPHONE XS', 'IPHONE 5C', 'ZTE Z233VL',
           'ZTE Z963VL', 'LG 235C', 'ALCATEL A577VL', 'LG L44VL',
           'SAMSUNG S975L', 'LG L61AL', 'LG L83BL', 'SAMSUNG G955U',
           'LG L62VL', 'LG L64VL', 'MOTOROLA XT1920DL', 'LG231C', 'iPhone X',
           'SAMSUNG S902L', 'LG 238C', 'SAMSUNG S550TL', 'LG L39C',
           'ZTE Z289L', 'SAMSUNG S920L', 'ZTE Z837VL', 'LG 236C', 'LG 440G',
           'ZTE Z558VL', 'LG L81AL', 'IPHONE 6 PLUS', 'iPhone 11 Pro Max',
           'SAMSUNG S765C', 'iPhone 11', 'SAMSUNG S757BL', 'LG108C',
           'SAMSUNG S380C', 'MOTOROLA XT1952DL', 'HUAWEI H883G',
           'ALCATEL A501DL', 'IPHONE XS MAX', 'Samsung T528G', 'LG L43AL',
           'ALCATEL A450TL', 'ZTE Z795G', 'LG L16C', 'iPhone 12',
           'SAMSUNG S906L', 'BRING YOUR TABLET', 'FRANKLIN WIRELESS F900HSVL',
           'SAMSUNG S336C', 'iPhone 11 Pro', 'ZTE Z716BL', 'LG LML211BL',
           'LG L57BL', 'LG220C', 'LG L21G', 'LG L33L', 'LG L163BL',
           'SAMSUNG S730G', 'MOTOROLA INC', 'LG L53BL', 'ZTE Z936L',
           'Samsung R451C', 'ZTE Z288L', 'ALCATEL A574BL', 'ALCATEL A621BL',
           'ZTE Z557BL', 'ALCATEL A564C', 'ALCATEL A521L', 'LG L35G',
           'LG L22C', 'LG L157BL', 'Samsung N981U1', 'BLU B100DL', 'LG 306G',
           'SAMSUNG S890L', 'RELIANCE AX54NC', 'SAMSUNG S968C',
           'HUAWEI H210C', 'LG 221C', 'HUAWEI H892L', 'LG L51AL', 'ZTE Z796C',
           'ZTE Z930L', 'ZTE Z719DL', 'ALCATEL A571VL', 'ALCATEL A392G',
           'Alcatel A508DL', 'LG 442BG', 'HUAWEI H715BL', 'ZTE Z986DL',
           'LG L31L', 'ZTE Z932L', 'ZTE Z916BL', 'Samsung G981U1',
           'NOKIA E5G', 'LG L59BL', 'SAMSUNG G973U1', 'Samsung G770U1',
           'SAMSUNG G975U1', 'LG L15G', 'LG L86C', 'LG 237C',
           'MOTOROLA W419G', 'iPhone 12 Pro Max', 'Samsung N975U1',
           'ZTE Z791G', 'ALCATEL A462C']

    phone_info['fm_radio'].fillna('is_miss',inplace = True)#fill with missing 
    phone_info['available_online'].fillna('is_miss',inplace=True)
    phone_info['device_type'].fillna('is_miss',inplace=True)
    phone_info['device_type'].replace(['M2M','BYOT','MOBILE_BROADBAND','FEATURE_PHONE','WIRELESS_HOME_PHONE'], ['Others','Others','Others','Others','Others'], inplace=True)
    phone_info['display_description'].replace(type_list,['Others']*len(type_list),inplace=True)
    phone_info['data_capable'].fillna(0.0,inplace=True)
    phone_info['device_lock_state'].fillna('is_miss',inplace=True)
    phone_info['device_lock_state'].replace(['LOCKED','UNLOCKED'], ['Others','Others'], inplace=True)
    phone_info['bluetooth'].fillna('is_miss',inplace = True)
    phone_info['battery_removable'].fillna('is_miss',inplace = True)

    new_phone_info = pd.get_dummies(phone_info,columns=['available_online','device_type','device_lock_state','data_capable','bluetooth','battery_removable','fm_radio','display_description'],drop_first=True)
    select_features = ['line_id', 'available_online_Y', 'available_online_is_miss',
           'device_type_Others', 'device_type_SMARTPHONE', 'device_type_is_miss',
           'device_lock_state_UNLOCKABLE', 'device_lock_state_is_miss',
           'data_capable_1.0', 'bluetooth_Y', 'bluetooth_is_miss',
           'battery_removable_Y', 'battery_removable_is_miss', 'fm_radio_Y',
           'fm_radio_is_miss', 'display_description_IPHONE 6S',
           'display_description_LG L722DL', 'display_description_Others',
           'display_description_Samsung S506DL', 'display_description_is_miss']
    new_phone_info = new_phone_info[select_features].drop_duplicates().reset_index(drop=True)
    new_phone_info.to_csv(root_folder + to_data_path + "new_phone_info.csv",header=True,index=None)

    #redemptions
    
    redemptions=pd.read_csv(data_path + "redemptions.csv")

    redemptions['red_count'] = scaler.fit_transform(redemptions.groupby('line_id')['channel'].transform('count').values.reshape(-1,1))#how ofen use
    redemptions['red_mean_rev'] = scaler.fit_transform(redemptions.groupby('line_id')['gross_revenue'].transform('mean').values.reshape(-1,1))#how much
    redemptions['channel_unique'] = scaler.fit_transform(redemptions.groupby('line_id')['channel'].transform('nunique').values.reshape(-1,1))#what kinds of channel
    redemptions['red_type_unique'] = scaler.fit_transform(redemptions.groupby('line_id')['redemption_type'].transform('nunique').values.reshape(-1,1))# what kinds of paid type
    redemptions['red_type_most_fre'] = redemptions['line_id'].map(redemptions.groupby('line_id')['redemption_type'].agg(lambda x: x.value_counts().idxmax()))#most frequently paid type
    redemptions['channel_most_fre'] = redemptions['line_id'].map(redemptions.groupby('line_id')['channel'].agg(lambda x: x.value_counts().idxmax()))#most frequently channel
    redemptions=pd.get_dummies(redemptions,columns=['red_type_most_fre','channel_most_fre'],drop_first=True)
    redemptions = pd.merge(upgrades,redemptions,how='left',on='line_id')
    lst = ['red_count','red_mean_rev','channel_unique','red_type_unique']
    redemptions[lst] = redemptions[lst].fillna(redemptions[lst].mean())
    redemptions.fillna(0,inplace=True)
    
    select_features = [e for e in redemptions.columns if e not in ['channel','gross_revenue','redemption_date','redemption_type','revenue_type']]
    new_redemptions = redemptions[select_features].drop_duplicates().reset_index(drop=True)
    new_redemptions.to_csv(root_folder+ to_data_path + "new_redemptions.csv",header=True,index=None)

    #deactivations
    deactivations=pd.read_csv(data_path + "deactivations.csv")

    deactivations['dea_times'] = scaler.fit_transform(deactivations.groupby('line_id')['deactivation_date'].transform('count').values.reshape(-1,1))
    deactivations['dea_reason_uni_counts'] = scaler.fit_transform(deactivations.groupby('line_id')['deactivation_reason'].transform('nunique').values.reshape(-1,1))
    deactivations['dea_most_fre_reason'] = deactivations['line_id'].map(deactivations.groupby('line_id')['deactivation_reason'].agg(lambda x: x.value_counts().idxmax()))

    temp_list = ['STOLEN','REMOVED_FROM_GROUP','MINCHANGE','STOLEN CREDIT CARD','DEVICE CHANGE INQUIRY','PORTED NO A/I','WN-SYSTEM ISSUED']
    deactivations['dea_most_fre_reason'].replace(temp_list,['Other']*len(temp_list),inplace=True)
    deactivations=pd.get_dummies(deactivations,columns=['dea_most_fre_reason'])#not useing drop first for merging the upgrade line_id and fillna with 0

    new_deactivation = pd.merge(upgrades,deactivations,how='left',on = 'line_id')
    new_deactivation['dea_times'].fillna(-1,inplace=True)#have no deactivation record
    new_deactivation.fillna(0,inplace=True)#have no deactivation record

    select_features = ['line_id', 'dea_times',
           'dea_reason_uni_counts', 'dea_most_fre_reason_Other',
           'dea_most_fre_reason_PASTDUE', 'dea_most_fre_reason_PORT OUT',
           'dea_most_fre_reason_RISK ASSESSMENT', 'dea_most_fre_reason_UPGRADE','dea_most_fre_reason_CUSTOMER REQD']
    new_deactivation = new_deactivation[select_features].drop_duplicates().reset_index(drop=True)
    new_deactivation.to_csv(root_folder+ to_data_path+ "new_deactivation.csv",header=True,index=None)

    #suspensions
    suspensions=pd.read_csv(data_path + "suspensions.csv")
    suspensions['sus_count'] = scaler.fit_transform(suspensions.groupby('line_id')['suspension_start_date'].transform('count').values.reshape(-1,1))
    new_suspension = pd.merge(upgrades,suspensions,how='left',on = 'line_id')
    new_suspension['sus_count'].fillna(-1,inplace=True)
    select_features = ['line_id','sus_count']
    new_suspension = new_suspension[select_features].drop_duplicates().reset_index(drop=True)
    new_suspension.to_csv(root_folder+ to_data_path +"new_suspension.csv",header=True,index=None)

    #network_usage
    network_usage_domestic=pd.read_csv(data_path + "network_usage_domestic.csv")

    network_usage_domestic['network_used_day'] = network_usage_domestic.groupby('line_id')['date'].transform('count')

    features = ['line_id', 'hotspot_kb', 'mms_in', 'mms_out', 'sms_in',
           'sms_out', 'total_kb', 'voice_count_in', 'voice_count_total',
           'voice_min_in', 'voice_min_out']
    new_features = ['mean_hotspot_kb', 'mean_mms_in', 'mean_mms_out', 'mean_sms_in',
           'mean_sms_out', 'mean_total_kb', 'mean_voice_count_in', 'mean_voice_count_total',
           'mean_voice_min_in', 'mean_voice_min_out']

    temp = network_usage_domestic[features].groupby('line_id').transform('mean')
    temp.columns = new_features

    new_network_usage_domestic = pd.concat((network_usage_domestic,temp),axis=1)

    new_features = ['network_used_day','mean_hotspot_kb', 'mean_mms_in', 'mean_mms_out', 'mean_sms_in',
           'mean_sms_out', 'mean_total_kb', 'mean_voice_count_in', 'mean_voice_count_total',
           'mean_voice_min_in', 'mean_voice_min_out']

    new_network_usage_domestic[new_features] = scaler.fit_transform(new_network_usage_domestic[new_features])

    #merge upgrade id_line and fill with mean
    new_network_usage_domestic = pd.merge(upgrades,new_network_usage_domestic,how='left',on='line_id')
    features = ['network_used_day','mean_hotspot_kb', 'mean_mms_in', 'mean_mms_out', 'mean_sms_in',
           'mean_sms_out', 'mean_total_kb', 'mean_voice_count_in', 'mean_voice_count_total',
           'mean_voice_min_in', 'mean_voice_min_out']
    new_network_usage_domestic[features] = new_network_usage_domestic[features].fillna((new_network_usage_domestic[features].mean()))

    #populate table
    select_features = ['line_id','network_used_day','mean_hotspot_kb', 'mean_mms_in', 'mean_mms_out', 'mean_sms_in',
           'mean_sms_out', 'mean_total_kb', 'mean_voice_count_in', 'mean_voice_count_total',
           'mean_voice_min_in', 'mean_voice_min_out']
    new_network_usage_domestic = new_network_usage_domestic[select_features].drop_duplicates().reset_index(drop=True)
    new_network_usage_domestic.to_csv(root_folder+ to_data_path + "new_networ_usage_domestic.csv",header=True,index=None)
    print('Finished.')
    
#merge all tables
def merge_tables(f_type,name):
    """
    merge the tables, f_type: 'dev' or 'eval'
    must create new_data folder under your working folder
    """
    print('Start to merge...')
    data_path = name + '/new_data/' + f_type + '_'

    new_redemptions = pd.read_csv(root_folder+ data_path + "new_redemptions.csv")
    new_phone_info = pd.read_csv(root_folder+ data_path + "new_phone_info.csv")
    new_customer_info = pd.read_csv(root_folder+ data_path +"new_customer_info.csv")
    new_suspension = pd.read_csv(root_folder+ data_path +"new_suspension.csv")
    new_deactivation = pd.read_csv(root_folder+data_path + "new_deactivation.csv")
    new_network_usage_domestic = pd.read_csv(root_folder+ data_path +"new_networ_usage_domestic.csv")
    upgrades=pd.read_csv(data_folder+"data/" + f_type + "/upgrades.csv")

    table_list = [new_redemptions,new_phone_info,new_customer_info,new_suspension,new_deactivation,new_network_usage_domestic,upgrades]
    final_merge = pd.concat(table_list, join='inner', axis=1)
    final_merge = final_merge.loc[:,~final_merge.columns.duplicated()]
    final_merge.to_csv(root_folder + data_path + "final_merge.csv",header=True,index=None)
    print('Finished')

#call above functions
def main(f_type,name):
    create_new_data(f_type,name)
    merge_tables(f_type,name)
