from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window

class Transform:
    def __init__(self, df, time_col, transform_dict, cross_section_col=None, normalization=None, fill_value=True, other_dict=False):
        self.time_col = time_col
        self.cross_section_col = cross_section_col
        if cross_section_col == None:
            self.cross_section = False
            variables = [cols for cols in df.columns if cols != time_col]
        else:
            self.cross_section = True
            self.window_spec = Window.partitionBy(cross_section_col)
            variables = [cols for cols in df.columns if cols != time_col and cols != cross_section_col]
        transformed_df = df.select(time_col, cross_section_col) if cross_section_col is not None else df.select(time_col)
        for cols in variables:
            if cols in transform_dict.keys():
                temp = df.select(time_col, cross_section_col, cols) if cross_section_col is not None else df.select(time_col)
                if 'time_split' in transform_dict[cols].keys():
                    temp = self.time_split(temp, cols, transform_dict[cols]['time_split'])
                
                if 'time_offset' in transform_dict[cols].keys():
                    temp = self.time_offset(temp, cols, transform_dict[cols]['time_offset'])

                if 'time_window' in transform_dict[cols].keys():
                    temp = self.time_window(temp, cols, transform_dict[cols]['time_window'])

                if 'saturation' in transform_dict[cols].keys():
                    temp = self.saturation(temp, cols, transform_dict[cols]['saturation'])
                
                if 'normalization' in transform_dict[cols].keys():
                    temp = self.normalize(temp, cols, transform_dict[cols]['normalization'])
                
                if normalization != None:
                    if 'normalization' not in transform_dict[cols].keys() and 'saturation' not in transform_dict[cols].keys():
                        temp = self.normalize(temp, cols, normalization_method=normalization)

                temp = temp.drop(time_col, cross_section_col)
                column_list = list(temp.columns)
                column_list.remove(cols)
                for entry in column_list:
                    temp = temp.withColumnRenamed(entry, cols+'['+entry+']')
            else: 
                temp = df.select(cols)           
            transformed_df = self.join_df(transformed_df, temp)
        self.output = transformed_df.orderBy(time_col)

    def time_split(self, data, column, date):
        data = data.withColumn('split_a' + date, F.when(F.col(self.time_col) >= date, F.col(column)).otherwise(0))
        data = data.withColumn('split_b' + date, F.when(F.col(self.time_col) < date, F.col(column)).otherwise(0))
        return data
    
    def time_offset(self, data, column, date):
        data = data.withColumn('offset_' + date, F.when(F.col(self.time_col) > date, F.col(column)).otherwise(0))
        return data
    
    def time_window(self, data, column, param_list):
        start_date = param_list[0]
        end_date = param_list[1]
        data = data.withColumn('window_'+start_date+'_'+end_date, F.when((F.col(self.time_col) >= start_date) & (F.col(self.time_col) <= end_date), F.col(column)).otherwise(0))
        return data

    def normalize(self, data, column, normalization_method='unit mean'):
        column_list = list(data.columns)
        column_list.remove(self.time_col)
        if self.cross_section:
            column_list.remove(self.cross_section_col)
        num_of_cols = len(column_list)
        if num_of_cols == 1:
            if normalization_method == 'zero mean':
                data = self.normalize_zero_mean(data, column)
                data = data.withColumnRenamed('transformed', 'nzm')
            else:
                data = self.normalize_unit_mean(data, column)
                data = data.withColumnRenamed('transformed', 'num')
        else:
            column_list.remove(column)
            for cols in column_list:
                if normalization_method == 'zero mean':
                    data = self.normalize_zero_mean(data, cols)
                    data = data.withColumnRenamed('transformed', cols+':nzm')
                    data = data.drop(cols)
                else:
                    data = self.normalize_unit_mean(data, cols)
                    data = data.withColumnRenamed('transformed', cols+':num')
                    data = data.drop(cols)
        return data
    
    def normalize_zero_mean(self, data, column_name):
        if self.cross_section:
            data = data.withColumn('transformed', F.round((F.col(column_name)-F.mean(F.when(F.col(column_name)!=0, F.col(column_name)).otherwise(None)).over(self.window_spec))/F.round(F.stddev(F.when(F.col(column_name)!=0, F.col(column_name)).otherwise(None)).over(self.window_spec),2),2))
        else:
            data = data.withColumn('transformed', F.round((F.col(column_name)-F.mean(F.when(F.col(column_name)!=0, F.col(column_name)).otherwise(None)))/F.round(F.stddev(F.when(F.col(column_name)!=0, F.col(column_name)).otherwise(None)),2),2))
        return data

    def normalize_unit_mean(self, data, column_name):
        if self.cross_section:
            data = data.withColumn('transformed', F.round(F.col(column_name)/F.mean(F.when(F.col(column_name)!=0, F.col(column_name)).otherwise(None)).over(self.window_spec),2))
        else:
            data = data.withColumn('transformed', F.round(F.col(column_name)/F.mean(F.when(F.col(column_name)!=0, F.col(column_name)).otherwise(None)),2))
        return data
    
    def saturation(self, data, column, param_dict):
        column_list = list(data.columns)
        column_list.remove(self.time_col)
        if self.cross_section:
            column_list.remove(self.cross_section_col)
        num_of_cols = len(column_list)
        range_of_k = self.array_of_transform(param_dict['k'])
        range_of_s = self.array_of_transform(param_dict['s'])
        if num_of_cols == 1:
            for k in range_of_k:
                for s in range_of_s:
                    data = self.saturation_logic(data, column, k, s)
                    data = data.withColumnRenamed('transformed', f'sat_k{k}s{s}')
        else:
            column_list.remove(column)
            for cols in column_list:
                for k in range_of_k:
                    for s in range_of_s:
                        data = self.saturation_logic(data, cols, k, s)
                        data = data.withColumnRenamed('transformed', cols+f'sat_k{k}s{s}')
                        data = data.drop(cols)
        return data
    
    def saturation_logic(self, data, column_name, k, s):
        k = round(k,1)
        temp_num = self.normalize_unit_mean(data, column_name)
        if self.cross_section:
            data = temp_num.withColumn('transformed', F.round(1 / (1 + (k/F.col('transformed'))**s),2))
        else:
            data = temp_num.withColumn('transformed', F.round(1 / (1 + (k/F.col('transformed'))**s),2))
        return data
    
    def join_df(self, df1, df2):
        temp1 = df1.withColumn("id", F.monotonically_increasing_id())
        temp2 = df2.withColumn("id", F.monotonically_increasing_id())
        temp = temp1.join(temp2, "id", "inner").drop("id")
        return temp

    def array_of_transform(self, list_of_transform):
        if len(list_of_transform) <= 2:
            range_of_values = list_of_transform[:]
        else:
            range_of_values = list(range(list_of_transform[0], list_of_transform[1] + 1, list_of_transform[2]))
        return range_of_values

if __name__ == '__main__':
    spark = SparkSession.builder.getOrCreate()
    raw_df = spark.read.csv('sample.csv', header=True, inferSchema=True)
    transform_dict = {'display_imp': {'adstock': [10, 30, 10]}}
    output = Transform(raw_df, time_col='date', cross_section_col='dma', transform_dict=transform_dict)
    print(output.output.columns)
    output.output.show()