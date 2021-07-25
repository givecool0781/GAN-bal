__author__ = 'CEH'
__data__ = 'XXXXX'
import numpy as np
import pandas as pd



 # reads csv file and returns np array of X,y -> of shape (N,D) and (N,1)
def load_data():
    data=pd.read_csv('CIC_all_add_malflow.csv')
    num_records,num_features = data.shape
    print("there are {} flow records with {} feature dimension".format(num_records,num_features))
    # there is white spaces in columns names e.g. ' Destination Port'
    # So strip the whitespace from  column names
    data = data.rename(columns=lambda x: x.strip())
    print('stripped column names')
    df_label = data['Label']
#         df_nor = data.loc(data['Label']=='BENIGN')
#         df_abnor = data.loc(data['Label']!=0)
#         df_nor.sample(frac= 0.5, replace= True, random_state=2)
#         data=pd.concat([df_nor, df_abnor],ignore_index = True)
    data = data.drop(columns=['Flow Packets/s','Flow Bytes/s','Label']) #CIC dropped bad columns & Label
#         print(data.shape)

    print('dropped bad columns')
    
    nan_count = data.isnull().sum().sum()
    print('There are {} nan entries'.format(nan_count))
    
    if nan_count>0:
        data.fillna(data.mean(), inplace=True)
        print('filled NAN')
    # data = data.drop(columns=col)
    print('remove bad features')
    data = data.astype(float).apply(pd.to_numeric)
    print('converted to numeric')
    
    # lets count if there is NaN values in our dataframe( AKA missing features)
    assert data.isnull().sum().sum()==0, "There should not be any NaN"
    
    
    X = data.values
    y = encode_label(df_label.values)
    return (X,y)


        
#We balance data as follows:
#1) oversample small classes so that their population/count is equal to mean_number_of_samples_per_class
#2) undersample large classes so that their count is equal to mean_number_of_samples_per_class
def balance_data(X,y,seed,datatype):
    np.random.seed(seed)
    unique,counts = np.unique(y,return_counts=True)
#     y1 = pd.DataFrame(y)
#     count_nor = y1.count(0)
    if datatype == 'KDD':
        counts_weight = np.array([1,1,1,1,1])
    elif datatype == 'CICIDS':
        counts_weight = np.array([0,1.1,0,1,1,1,0.1,0.001,0.001,0.0010]) #CIC weight 
    elif datatype == 'UNSW':
        counts_weight = np.array([1,1,1,1,1,1,0.1,1,1,1]) #CIC weight
    elif datatype == 'XXX':
        counts_weight = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]) #CIC weight
    elif datatype == 'CICIDS_dup':
        counts_weight = np.array([0,0,10,10,10,10,10,10,10,10]) #CIC weight
#     counts_weight = np.array([1,1])
#     mean_samples_per_class = int(round(np.mean(counts-count_nor)))
    mean_samples_per_class = int(round(np.average(counts,weights = counts_weight))) # 取平均時先排除正常流量
    N,D = X.shape #(number of examples, number of features)
    new_X = np.empty((0,D)) 
    new_y = np.empty((0),dtype=int)
    for i,c in enumerate(unique):
        temp_x = X[y==c]
        indices = np.random.choice(temp_x.shape[0],mean_samples_per_class) # gets `mean_samples_per_class` indices of class `c`
        new_X = np.concatenate((new_X,temp_x[indices]),axis=0) # now we put new data into new_X 
        temp_y = np.ones(mean_samples_per_class,dtype=int)*c
        new_y = np.concatenate((new_y,temp_y),axis=0)
        
    # in order to break class order in data we need shuffling
    indices = np.arange(new_y.shape[0])
    np.random.shuffle(indices)
    new_X =  new_X[indices,:]
    new_y = new_y[indices]
    return (new_X,new_y)


# chganges label from string to integer/index
def encode_label(Y_str):
    labels_d = make_value2index(np.unique(Y_str))
    Y = [labels_d[y_str] for y_str  in Y_str]
    Y = np.array(Y)
    return np.array(Y)


def make_value2index(attacks):
    #make dictionary
    attacks = sorted(attacks)
    d = {}
    counter=0
    for attack in attacks:
        d[attack] = counter
        counter+=1
    return d


# normalization
def normalize(data):
        data = data.astype(np.float32)
       
        eps = 1e-15

        mask = data==-1
        data[mask]=0
        mean_i = np.mean(data,axis=0)
        min_i = np.min(data,axis=0) #  to leave -1 (missing features) values as is and exclude in normilizing
        max_i = np.max(data,axis=0)

        r = max_i-min_i+eps
        data = (data-mean_i)/r  # zero centered 

        #deal with missing features -1
        data[mask] = 0        
        return data