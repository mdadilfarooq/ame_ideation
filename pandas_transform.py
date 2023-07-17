import pandas as pd

class Transform:
    def __init__(self, df, time_col, transform_dict, cross_section_col=None, normalization=None, fill_value=True, other_dict=False):
        if cross_section_col == None:
            cross_section = False
            df=df.copy()
            variables=[col for col in df.columns if col != time_col]
        else:
            cross_section = True
            df=df.set_index(cross_section_col)
            variables=[col for col in df.columns if col != time_col and col != cross_section_col]

        transformed_df = pd.DataFrame()

        for col in variables:
            if col in transform_dict.keys():
                temp = pd.DataFrame()
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
                
                if normalization != None: 
                    if 'normalization' not in transform_dict[col].keys():
                        if 'saturation' not in transform_dict[col].keys():
                            temp=self.normalize_df(temp,normalization_method=normalization,cross_section=cross_section)

                temp=temp.add_prefix(col+' [')
                temp=temp.add_suffix(']') 

                temp=pd.concat([df[col],temp],axis=1)
                transformed_df=pd.concat([transformed_df,temp],axis=1)

            else:
                temp=pd.DataFrame()
                temp[col]=df[col]
                transformed_df=pd.concat([transformed_df,temp],axis=1)

        transformed_df=pd.concat([df[time_col],transformed_df],axis=1)
        transformed_df.reset_index(inplace=True)
        self.output_df = transformed_df.sort_values([time_col,cross_section_col])

    def normalize_unit_mean(self, data, cross_section=True):
        if cross_section==True:
            data_norm=round(data.groupby(data.index.names).transform(lambda x: (x/x[x!=0].mean())),2)
        else:
            data_norm=round(data.transform(lambda x: (x/x[x!=0].mean())),2)
        return data_norm


    def normalize_zero_mean(self, data, cross_section=True):
        if cross_section==True:
            data_norm=round(data.groupby(data.index.names).transform(lambda x: (x-x[x!=0].mean())/round(x[x!=0].std(),2)),2)
        else:
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
    
    def saturation(self,x,k,s,cross_section=True):
        k=round(k,1)
        x_num = self.normalize_unit_mean(x)
        if cross_section==True:
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
                temp['sat_k{}s{}'.format(k,s)]=self.saturation(x,k,s,cross_section)
        return temp

    def saturation_df(self,x,param_dict,cross_section=True):
        temp=pd.DataFrame()
        range_of_k=self.array_of_transform(param_dict['k'])
        range_of_s=self.array_of_transform(param_dict['s'])
        for i in x.columns:
            for k in range_of_k:
                for s in range_of_s:
                    temp[i+'sat_k{}s{}'.format(k,s)]=self.saturation(x[i],k,s,cross_section)
        return temp

    def array_of_transform(self, list_of_transform):
        if len(list_of_transform) <= 2:
            range_of_values = list_of_transform[:]
        else:
            range_of_values = list(range(list_of_transform[0], list_of_transform[1] + 1, list_of_transform[2]))
        return range_of_values
    
if __name__ == '__main__':
    raw_df = pd.read_csv('sample.csv')
    transform_dict = {'web_bookings': {'normalization': 'zero mean'}}
    output = Transform(raw_df,time_col='date',cross_section_col='dma',transform_dict=transform_dict)
    print(output.output_df.columns)
    print(output.output_df.head(20))
    