import pandas as pd
import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import StandardScaler
# import datetime

class Transform:

    def __init__(self,df, time, transform_dict,cross_section_col=None, 
               normalization=None,fill_value = True,other_dict=False):
        """Parameters:
        1. df: Dataframe
            the modeling dataframe that consists of raw variables for transformation.

        2. time: string
            Name of the timestamp variable in the dataframe.

        3. cross_section: string
            Name of the region (dma/zip code/city/state) variable

        4. transform_dict: dictionary
            This contains the variables and the transformations to be performed on them.
            All the transformations to be performed (single/multiple) should be provided in the form of a list.

        5. normalization: string
            specifying the kind of normalization to be performed.

            Default:'unit mean': all the values in a given cross-section are scaled to have an average of 1.
            Other option: 'zero mean': all the values in a given cross-section are scaled to have an average of 0.

        6. fill_value: boolean
            specifying whether the N/A values created by lagged values have to be filled with 0 or the first value of the series.
            Default: True: the first value of the series will be used to populate the null valus created due to lagging
            Other option: False: 0 will be used to populate the null values.

        Returns: Dataframe
            Includes all the variables from the orignal dataframe and the transformed variables, as specified in the dictionary

        Notes:
        For every variable that is not present in the dictionary, the function returns the variable and the normalized value of the variable.
        """
        if cross_section_col == None:
            cross_section=False
            df=df.copy()
            variables=[col_names for col_names in df.columns if col_names != time]
        else:
            cross_section=True
            df=df.set_index(cross_section_col)
            #creating a list of all variables in the dataframe (minus the time and cross-section column)
            variables=[col_names for col_names in df.columns if col_names != time and col_names != cross_section]

        # dataframe to store final results
        ame_data = pd.DataFrame()

        for col in variables:
            #iterating over all the variables in the dataframe.
            #if any variable is present in the dictionary, then it will run through this loop checking if any one or multiple\
            #transformations are to be performed.
            if col in transform_dict.keys():
                temp=pd.DataFrame()
                if 'time_split' in transform_dict[col].keys():
                    temp=self.time_splitter_series(df,col,time,transform_dict[col]['time_split'])
                if 'time_offset' in transform_dict[col].keys():
                    temp=self.time_offset_series(df,col,time,transform_dict[col]['time_offset'])
                if 'time_window' in transform_dict[col].keys():
                    temp=self.time_window_series(df,col,time,transform_dict[col]['time_window'][0],transform_dict[col]['time_window'][1])
               
            
                if 'adstock' in transform_dict[col].keys():
                    if temp.shape[1]==0:
                        temp=self.adstock_series(df[col],transform_dict[col]['adstock'], cross_section=cross_section)
                    else:
                        temp=self.adstock_df(temp,transform_dict[col]['adstock'], cross_section=cross_section)
                #for the remaining transformations, an if-else condition has been applied to check if the variable has been transformed.
                # if yes, then temp.shape[1]>0 and the else condition is executed.
                # if not, the first function is executed.
                #every function creates a new dataframe and stores the result in that dataframe
                #this removes the necessity of dropping columns as we progress throughout the code.
            
                if 'lag' in transform_dict[col].keys():
                    if temp.shape[1]==0:
                        temp=self.lag_series(df[col],transform_dict[col]['lag'],cross_section =cross_section,fill_value=fill_value)
                    else:
                        temp=self.lag_df(temp,transform_dict[col]['lag'],cross_section =cross_section,fill_value=fill_value)
            
                if 'moving_avg' in transform_dict[col].keys():
                    if temp.shape[1]==0:
                        temp=self.moving_avg_series(df[col], transform_dict[col]['moving_avg'], cross_section=cross_section)
                    else:
                        temp=self.moving_avg_df(temp,transform_dict[col]['moving_avg'],cross_section=cross_section)
              
                if 'weibull' in transform_dict[col].keys():
                    if temp.shape[1]==0:
                        temp=self.weibull_series(df[col], transform_dict[col]['weibull'], cross_section=cross_section)
                    else:
                        temp=self.weibull_df(temp,transform_dict[col]['weibull'],cross_section=cross_section)
            
                if 'saturation' in transform_dict[col].keys():
                    if temp.shape[1]==0:
                        temp=self.saturation_series(df[col],transform_dict[col]['saturation'], cross_section=cross_section)
                    else:
                        temp=self.saturation_df(temp,transform_dict[col]['saturation'],cross_section=cross_section)
                    
                if 'normalization' in transform_dict[col].keys():
                    if temp.shape[1]==0:
                        temp=self.normalize_series(df[col],transform_dict[col]['normalization'],cross_section=cross_section)
                    else:
                        temp=self.normalize_df(temp,transform_dict[col]['normalization'],cross_section=cross_section) 
                #this normalization pertains to the normalization defined in the transform()
                #while the above pertains to whether normalization was specified inside the dictionary.
                if normalization != None: 
                    if 'normalization' not in transform_dict[col].keys():
                        if 'saturation' not in transform_dict[col].keys():
                            temp=self.normalize_df(temp,normalization_method=normalization,cross_section=cross_section)
                #adding the name of the variable and enclosing the transformations within square brackets.
                temp=temp.add_prefix(col+' [')
                temp=temp.add_suffix(']') 
                #adding the non-transformed variable as well and storing them in the final dataframe: ame_data
                temp=pd.concat([df[col],temp],axis=1)
                ame_data=pd.concat([ame_data,temp],axis=1)
            else:
                #if columns are not there in the dictionary, we want them as it is
                temp=pd.DataFrame()
                temp[col]=df[col]
                ame_data=pd.concat([ame_data,temp],axis=1)
            
        #adding the time variable to the final dataframe
        ame_data=pd.concat([df[time],ame_data],axis=1)
        #reseting index
        ame_data.reset_index(inplace=True)
        self.output_df = ame_data

    def read_df(self,path, sheet_name = None):
        """
        Reads a dataframe from the specified file name. Supports both excel files and csv

        :param: path: The path of the file or a pandas dataframe
        :param: sheet_name: The sheet name if xlsx was passed
        :returns: A dataframe extracted from the specified file
        """
        if isinstance(path, pd.DataFrame):
            return path
        if ".csv" in path:
            df = pd.read_csv(path)
        elif '.xlsx' in path:
            df = pd.read_excel(path, sheet_name = sheet_name, engine='openpyxl')
        else:
            assert False, "Unsupported File Type"
        return df

    def normalize_unit_mean(self, data, cross_section=True):
        '''Parameters:
        1. data: series
            the time series that has to be normalized.

        Returns: series
            time series scaled to an average of 1'''
        if cross_section == True:
            data_norm=round(data.groupby(data.index.names).transform(lambda x: (x/x[x!=0].mean())),2)
        else:
            #only using non zero variables for the calculation of mean
            data_norm=round(data.transform(lambda x: (x/x[x!=0].mean())),2)
        return data_norm


    def normalize_zero_mean(self, data, cross_section=True):
        '''Parameters:
        1. data: series
            the time series that has to be normalized.

        Returns: series
            time series scaled to an average of 0'''
        if cross_section ==True:
            data_norm=round(data.groupby(data.index.names).transform(lambda x: (x-x[x!=0].mean())/round(x[x!=0].std(),2)),2)
        else:
            #only using non zero variables for the calculation of mean and standard deviation.
            data_norm=round(data.transform(lambda x: (x-x[x!=0].mean())/round(x[x!=0].std(),2)),2)
        return data_norm

    def normalize_series(self,x,normalization_method='unit mean',cross_section=False):
        temp=pd.DataFrame()
        if normalization_method=='zero mean':
            temp['nzm']=self.normalize_zero_mean(x,cross_section =cross_section)
        else:
            temp['num']=self.normalize_unit_mean(x,cross_section =cross_section)
        return temp
    
    def normalize_df(self,x,normalization_method='unit mean',cross_section=False):
        temp=pd.DataFrame()
        for i in x.columns:
            if normalization_method=='zero mean':
                temp[i+':nzm']=self.normalize_zero_mean(x[i],cross_section =cross_section)
            else:
                temp[i+':num']=self.normalize_unit_mean(x[i],cross_section =cross_section)
        return temp  

    def adstock_series(self, x, adstock_rate_list, cross_section=False):
        '''
        Parameters:
        x: series
        k: adstock rate
        Returns:
        series with the adstock transformation.
        This function can handle different cross-sections of data.
        The cross-section has to be a part of the index.
        The function "transform()" (defined later in this notebook) sets the variable corresponding to the cross-section
        as the index of the dataframe.
        The function groups the data according to the cross-sections and calls the function "adstock_region()"
        This sub-function performs the adstock for every cross-section"'''
        temp=pd.DataFrame()
        range_of_adstock=self.array_of_transform(adstock_rate_list)
        for ad in range_of_adstock:
            if cross_section==True:
                #the column are dynamically named with 'ad' followed by the adstock rate
                temp['ad{}'.format(ad)]=x.groupby(x.index.names).apply(lambda x: self.adstock_region(x,ad))
            else:
                temp['ad{}'.format(ad)]=self.adstock_region(x,ad)
        return temp
        #this function is designed to adstock all the columns in a dataframe
        #this is used when multiple transformations have been performed and we need to adstock the transformed data
        #to keep into account the transformations already performed, the column names are appended with 'adstock' followed by the adstock value.

    def adstock_region(self, p, adstock_rate):
        '''
        This function will take 1 cross-section at a time and do the adstock transformation for that cross-section'''
        total_len=len(p)
        ad=pd.DataFrame()
        adstock=[]
        for i in range(0, total_len):
            if i==0:
                adstock.append(round(p.iloc[i],2))
            else:
                adstock.append(round(p.iloc[i] + (adstock_rate/100)*adstock[i-1],2))
        ad['new_value']=adstock
        ad.set_index(p.index, inplace=True)
        return ad.new_value

    def adstock_df(self,x,adstock_rate_list, cross_section=False):
        temp=pd.DataFrame()
        range_of_adstock=self.array_of_transform(adstock_rate_list)
        for i in x.columns:
            for ad in range_of_adstock:
                if cross_section==True:
                    #the column are dynamically named with 'ad' followed by the adstock rate
                    temp[i+':ad{}'.format(ad)]=x[i].groupby(x.index.names).apply(lambda x: self.adstock_region(x,ad))
                else:
                    temp[i+':ad{}'.format(ad)]=self.adstock_region(x[i],ad)
        return temp

    def lag(self,x, j, cross_section=True, fill_value=True):
        '''
        Input: list (can contain upto 3 elements)
        Enter the value for lag in the form of a list

        [1]: only a lag of 1 will be performed
        [1,3]: a lag of 1 and 3 will be performed
        [2,8,2]: lag from 2-8 in steps of 2 will be performed'''
        if cross_section ==True:
            lag_series=x.groupby(x.index.names).shift(j)
        else:
            lag_series=x.shift(j)

        if fill_value==True:
            if j>=0:
                lag_series.fillna(method='bfill',inplace=True)
            else:
                lag_series.fillna(method='ffill',inplace=True)
        else:
            lag_series.fillna(0, inplace=True)
        return lag_series 

    #this function is designed to lag all the columns in a dataframe
    #this is used when multiple transformations have been performed and we need to lag the transformed data
    #to keep into account the transformations already performed, the column names are appended with 'lag' followed by the lag value.
    def lag_df(self,x,list_of_lag, cross_section =True,fill_value=True):
        temp=pd.DataFrame()
        range_of_lag=self.array_of_transform(list_of_lag)
        for i in x.columns:
            for j in range_of_lag:
                temp[i+':lag{}'.format(j)]=self.lag(x[i],j,cross_section=cross_section,fill_value=fill_value)      
        return temp

    #this function is designed to lag a series
    #this is used when lag is the only transformation done
    #the column are named as:'lag' followed by the lag value.
    def lag_series(self,x,list_of_lag, cross_section =True,fill_value=True):
        temp=pd.DataFrame()
        range_of_lag=self.array_of_transform(list_of_lag)
        for j in range_of_lag:
            temp['lag{}'.format(j)]=self.lag(x,j,cross_section=cross_section,fill_value=fill_value)
        return temp

    def saturation(self,x,k,s):
        '''
        Parameters:
        x: series
    
        k: float
        k denotes the half-saturation point
    
        s: float
        s denotes the slope
    
        The variable is normalized initially before being transformed.
        '''

        k=round(k,1)
        x_num = self.normalize_unit_mean(x)
        if self.cross_section==True:
            x_sat=x_num.groupby(x_num.index.names).transform(lambda x: 1/(1+(k/x)**s))
        else:
            x_sat=x_num.transform(lambda x: 1/(1+(k/x)**s))
        return round(x_sat,2)


    def saturation_series(self,x,param_dict,cross_section=True):
        temp=pd.DataFrame()
        range_of_k=self.array_of_transform(param_dict['k'])
        range_of_s=self.array_of_transform(param_dict['s'])
        for k in range_of_k:
            for s in range_of_s:
                temp['sat_k{}s{}'.format(k,s)]=self.saturation(x,k,s)
        return temp

    def saturation_df(self,x,param_dict,cross_section=True):
        temp=pd.DataFrame()
        range_of_k=self.array_of_transform(param_dict['k'])
        range_of_s=self.array_of_transform(param_dict['s'])
        for i in x.columns:
            for k in range_of_k:
                for s in range_of_s:
                    temp[i+'sat_k{}s{}'.format(k,s)]=self.saturation(x[i],k,s)
        return temp   

    def weibull_region(self, campaign, window_param, shape_param):
        import math
        from collections import deque
        total_len = len(campaign)
        intermediate = pd.DataFrame()
        intermediate['data'] = campaign.values
        lambda_ = window_param / ((-np.log(0.001)) ** (1 / shape_param))

        a = {w :math.exp((-(w / lambda_) ** shape_param)) for w in range(0,window_param)}
        temp = deque([])
        new_col = []
        for value in range(0,total_len):
            temp.append(intermediate['data'][value]) 
            if len(temp) > window_param:
                temp.popleft()
            new_ele = 0
            for i in range(len(temp)):
                new_ele += (temp[len(temp) - i - 1] *  a[i])
            new_col.append(new_ele)
        new_col = pd.Series(new_col)
        new_col.index = campaign.index
        return new_col

    def weibull_series(self, x, param_dictionary, cross_section=False):
        temp = pd.DataFrame()
        range_of_window = self.array_of_transform(param_dictionary['window'])
        range_of_k = self.array_of_transform(param_dictionary['k'])
        for w in range_of_window:
            for k in range_of_k:
                if cross_section == True:
                    temp['wa_w{}k{}'.format(w,k)] = x.groupby(x.index.names).apply(lambda x: self.weibull_region(x, w, k))
                else:
                    temp['wa_w{}k{}'.format(w,k)] = self.weibull_region(x,w,k)
        return temp


    def weibull_df(self, x, param_dictionary, cross_section=False):
        temp = pd.DataFrame()
        range_of_window = self.array_of_transform(param_dictionary['window'])
        range_of_k = self.array_of_transform(param_dictionary['k'])
        for w in range_of_window:
            for k in range_of_k:
                for i in x.columns:
                    if cross_section == True:
                        temp[i + 'wa_w{}k{}'.format(w,k)] = x[i].groupby(x[i].index.names).apply(lambda x: self.weibull_region(x, w, k))
                    else:
                        temp[i + 'wa_w{}k{}'.format(w,k)] = self.weibull_region(x[i],w,k)
        return temp
        
    def ma_weights(self, weights):

        if len(weights) > 1:
            # to convert float weights into integer type
            weights = [int(100 * i) if type(i) == float else i for i in weights]
            #appending the last element in the list multiple times to match with the window length                    
            while len(weights) -1 < weights[0]:
                weights.append(weights[-1])
            #truncating the last elements untill we match with the length of the window
            while len(weights) -1 > weights[0]: 
                del weights[-1]

            weights1 = [i / sum(weights[1:]) for i in weights if i != weights[0]]
            weights1.insert(0, weights[0])
        # to handle uniform moving average
        else:
            weights1 = [1 / weights[0]] * weights[0]
            weights1.insert(0, weights[0])
        return weights1

    def moving_avg(self, x, weights, cross_section=True, center=False):
        #changing the weights in a particular format to give it to the function
        weights = self.ma_weights(weights)
        if cross_section == True:
            if center == False:
                moving_avg_weighted = pd.DataFrame(x.groupby(x.index.names).rolling(weights[0]).apply(\
                                        lambda x: np.dot(x, np.array(weights[1:])) / len(np.array(weights[1:])), raw=True))
                moving_avg_weighted = moving_avg_weighted.fillna(method='bfill',axis=0)  #backfilling the nan values
                moving_avg_weighted.reset_index(level=0, drop=True, inplace=True)
            else:
                moving_avg_weighted = pd.DataFrame(x.groupby(x.index.names).rolling(weights[0], center=True).apply(\
                                        lambda x: np.dot(x, np.array(weights[1:])) / len(np.array(weights[1:])), raw=True))
                moving_avg_weighted = moving_avg_weighted.groupby(x.index).fillna(method='bfill', axis=0) #backfilling the intial nan values of a column
                moving_avg_weighted = moving_avg_weighted.groupby(x.index).fillna(method='ffill', axis=0) #forwardfilling the last nan values of a column
                moving_avg_weighted.reset_index(level=0, drop=True, inplace=True)
        else:
            if center==False:
                moving_avg_weighted = pd.DataFrame(x.rolling(weights[0]).apply(\
                                        lambda x: np.dot(x, np.array(weights[1:])) / len(np.array(weights[1:])), raw=True))
                moving_avg_weighted= moving_avg_weighted.fillna(method='bfill',axis=0)
                moving_avg_weighted.reset_index(level=0, drop=True,inplace=True)
            else:
                moving_avg_weighted = pd.DataFrame(x.rolling(weights[0],center=True).apply(\
                                        lambda x: np.dot(x, np.array(weights[1:])) / len(np.array(weights[1:])), raw=True))
                moving_avg_weighted = moving_avg_weighted.groupby(x.index).fillna(method='bfill',axis=0)
                moving_avg_weighted = moving_avg_weighted.groupby(x.index).fillna(method='ffill',axis=0)
                moving_avg_weighted.reset_index(level=0, drop=True, inplace=True)

        return moving_avg_weighted

    def moving_avg_series(self, x, param_dict, cross_section=True):
        temp=pd.DataFrame()
        
        if 'f' in param_dict.keys():
            weights = param_dict['f']
            temp['ma_f'] = self.moving_avg(x, weights, cross_section=cross_section, center=False)
        if 'c' in param_dict.keys():
            weights = param_dict['c']
            temp['ma_c'] = self.moving_avg(x, weights, cross_section=cross_section, center=True)

        return temp

    def moving_avg_df(self, x, param_dict, cross_section=True):
        temp = pd.DataFrame()
        
        if 'f' in param_dict.keys():
            weights = param_dict['f']
            for i in x.columns:
                temp[i+'ma_f'] = self.moving_avg(x[i], weights, cross_section=cross_section, center=False)
        if 'c' in param_dict.keys():
            weights = param_dict['c']
            for i in x.columns:
                temp[i+'ma_c'] = self.moving_avg(x[i], weights, cross_section=cross_section, center=True)
        return temp


    def array_of_transform(self, list_of_transform):
        '''
        Parameters:
        list_of_transform: list
        
        Returns: list
        
        For every transformation, the user has the option to provide 1-3 inputs for the parameters.
        For instance, the user can give adstock:[10]/ adstock:[10,20]/ adstock: [20,50,10]
        This function has been created to handle different cases for variable transformations and avoid 
        repition inside the function: transform().
        This function can take a list as an input and generate a list of all the required transformations.
        
        If the user only wants 1 transformation: 
            input:[20] 
            output:[20]
        If the user wants 2 transformations: 
            input:[20,50] 
            output:[20,50]
        If the user wants multiple transformations from 20-50 in steps of 5:
            input:[20,50,5]
            output:[20,25,30,35,40,45,50]
        '''
        
        if len(list_of_transform) == 1:
            range_of_values = [list_of_transform[0]]

        elif len(list_of_transform) == 2:
            range_of_values = [list_of_transform[0],list_of_transform[1]]

        else:
            range_of_values = np.arange(list_of_transform[0], 
                                list_of_transform[1] + list_of_transform[2],
                                list_of_transform[2])
        
        return range_of_values   

    #these functions are called within the 'other dictionary' function
    def matching_keys(self, dict1, dict2):
        common_keys = set(dict1.keys()).intersection(set(dict2)) - set(['moving_avg', 'time_spliter', 'time_offset', 'time_window'])
        return common_keys

    def do_sorting(self, list1):
        list1.sort()
        return list1

    #this function has been defined to take 2 dictionaries: old and new
    #it will return the new dictionary updated with only those transformations which have not been performed in the old dictionary
    #this funciton has been designed to reduce the run time of the transform() function for multiple runs.

    def other_dictionary(self, test_dict1, test_dict):
        overlap_keys1 = self.matching_keys(test_dict1, test_dict)
        for i in overlap_keys1:
            overlap_keys2 = self.matching_keys(test_dict1[i], test_dict[i])
            for j in overlap_keys2:
                if isinstance(test_dict[i][j], list):
                    sorting = list(set(self.array_of_transform(test_dict[i][j])) - set(self.array_of_transform(test_dict1[i][j])))
                    test_dict[i][j] = self.do_sorting(sorting)
                if isinstance(test_dict[i][j],dict):
                    for k in self.matching_keys(test_dict[i][j], test_dict1[i][j]):
                        if isinstance(test_dict[i][j][k], list):
                            test_dict[i][j][k] = list(set(test_dict[i][j][k]) - set(test_dict1[i][j][k]))
        return test_dict

    def time_splitter_series(self, df, x, time, date):
        temp = pd.DataFrame()
        temp['all_data'] = df[x]
        temp1 = pd.DataFrame((np.where(df[time] >= date, df[x], 0)))
        temp.insert(temp.shape[1], 'split_a' + date, temp1)
        temp2 = pd.DataFrame(np.where(df[time] < date, df[x], 0))
        temp.insert(temp.shape[1], 'split_b'+ date, temp2)
        return temp

    def time_offset_series(self,df,x,time,date):
        temp = pd.DataFrame()
        temp['all_data'] = df[x]
        temp1 = pd.DataFrame(np.where(df[time]>date,df[x],0))
        temp.insert(temp.shape[1], 'offset_' + date, temp1)
        return temp

    def time_window_series(self,df,x,time,start_date,end_date):
        temp = pd.DataFrame()
        temp['all_data'] = df[x]
        temp1 = pd.DataFrame(np.where(df[time].between(start_date, end_date, inclusive=True), df[x], 0))
        temp.insert(temp.shape[1], 'window_' + start_date + '_' + end_date, temp1)
        return temp