from transform import Transform
# from model import Model
import pandas as pd

raw_data = pd.read_excel('Mock_Database.xlsx')

transform_dict = {'display_imp': {'time_split': '10-01-2021'}}

result=Transform(df=raw_data,
          time='date',
          cross_section_col='dma',
          transform_dict=transform_dict
         )

print(result.output_df)

# new_transform_dict = {'display_imp': {'adstock': [20, 80, 20], 'lag': [0, 8]},
#                   'meta_campaign1_click': {'adstock': [10, 80, 20], 'lag': [3, 8]},
#                   'meta_campaign2_click': {'adstock': [10, 50, 25], 'lag': [3, 8]},
#                   'paid_click': {'adstock': [20, 40, 10], 'lag': [5, 9]},
#                   'sales_ducati': {'adstock': [10, 5],
#                                   'lag': [3, 9]},
#                   'tv_grp': {'adstock': [40, 50], 'lag': [2, 12]}}

# bounds = {'app_bookings': {'lower_bound': -100000.0,
#                   'upper_bound': 100000.0,
#                   'value': 99645.7443},
#           'display_imp': {'lower_bound': -100000.0,
#                           'upper_bound': 100000.0,
#                           'value': -4721.83054},
#           'meta_campaign1_click': {'lower_bound': -100000.0,
#                                    'upper_bound': 100000.0,
#                                    'value': -87715.7938},
#           'meta_campaign2_click': {'lower_bound': -100000.0,
#                                    'upper_bound': 100000.0,
#                                    'value': 14773.8549},
#           'paid_click': {'lower_bound': -100000.0,
#                          'upper_bound': 100000.0,
#                          'value': -37305.6779},
#           'tv_grp': {'lower_bound': -100000.0,
#                      'upper_bound': 100000.0,
#                      'value': -30687.7499},
#           'web_bookings': {'lower_bound': -100000.0,
#                            'upper_bound': 100000.0,
#                            'value': -65952.7876}}

# m = Model(df = result.output_df,
#          target = 'sales_ducati',
#          col_json = bounds)

# m.outputs
# m.contribution()