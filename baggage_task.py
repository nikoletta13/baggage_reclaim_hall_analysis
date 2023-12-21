# baggage_task.py
"""
File to assist in the analysis of baggage claim hall task. 
"""									

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

############
# Read bag data
############

                                        
bags_df = pd.read_excel('Interview task.xlsx', sheet_name='2.1. Bags on the belt')												
# do not parse dates straight away - assumption is that after a certain point
# the conversion of percentage of day passed to time didn't execute. Further
# inspection on the exact values suggests that. In a real life situation I'd contact
# the team who carried out this conversion to verify my guess. In this case, as it is 
# a simulation, I'm assuming it is the case. 


def percent_to_time(p):
    p = float(p)
    p*=24
    hrs = int(p)%24   
    mins = int((p-int(p))*60) + (((p-int(p))*60)%1 >=0.5)
    try:
        dt = datetime.time(hour=hrs,minute=mins,second=0)
    except:
        input(p) 
    return dt

def time_to_percent(t:datetime.time):
    mins_total = t.hour*60+t.minute
    p = mins_total/(24*60)
    return p

# convert all to percentage of time passed
fixed_time = [None] * len(bags_df)
percentage_of_time = [None] * len(bags_df)
for i in range(len(bags_df)):
    if type(bags_df['Time'].iloc[i]) != datetime.time:
        fixed_time[i] = percent_to_time(bags_df['Time'].iloc[i])
        percentage_of_time[i] = bags_df['Time'].iloc[i]
    else:
        fixed_time[i] = bags_df['Time'].iloc[i]    
        percentage_of_time[i] = time_to_percent(bags_df['Time'].iloc[i])

bags_df['Time'] = fixed_time
bags_df['Percentage Time'] = percentage_of_time

sns.set_theme(style='darkgrid')
tt = [0,0.25,0.50, 0.75,0.999]
ll = [percent_to_time(x) for x in tt]
def timeplot_from_percent(df,y_p,hue=None,title=None,legend=None):
    if hue!= None:
        ax = sns.lineplot(data=df,x='Percentage Time',y=y_p, hue=hue)
    elif legend:
        ax = sns.lineplot(data=df,x='Percentage Time',y=y_p, legend=True)
    else:
        ax = sns.lineplot(data=df,x='Percentage Time',y=y_p)
    ax.set_xticks(ticks=tt, labels=ll)
    ax.set_xlabel('Time')
    if title:
        ax.set_title(title)
    plt.show()


# visualise rough
#-----------------
for i in range(1,5):
    timeplot_from_percent(bags_df[bags_df['Processor']==f'BELT {i}'],y_p = 'Bags',title=f'Bags on Belt {i}')

g = sns.relplot(data=bags_df, x='Percentage Time', y='Bags', row='Processor', hue='Processor',legend=True,kind='line')
sns.despine(left=True, bottom=True)
for item, ax in g.axes_dict.items():
    ax.set_xticks(ticks=tt, labels=ll)
    ax.set_title('')  

timeplot_from_percent(bags_df,y_p='Bags',hue='Processor',title = 'Amount of bags on each belt')
#-----------------


# total number of bags at any time
#-----------------

tot_df = bags_df.groupby('Percentage Time').sum('Bags')
print(f'Maximum number of bags at any time: {tot_df['Bags'].max()}')	
												

timeplot_from_percent(df = tot_df,y_p='Bags',title='Total amount of bags on the belts')
#-----------------

############
# Read passenger data
############


def minute_diff(dt_start,dt_end):
    d = dt_end.hour*60+dt_end.minute+dt_end.second/60 - (dt_start.hour*60+dt_start.minute+dt_start.second/60)
    return d

passengers_df = pd.read_excel('Interview task.xlsx', sheet_name='2.2. Passengers waiting', dtype={'Start Time':str, 'End Time' : str})	
# some dates have creeped into the times - read times as strings to clean


# remove that date
#-----------------
passengers_df['Start Time'] = passengers_df['Start Time'].apply(lambda x: x[-8:])
passengers_df['End Time'] = passengers_df['End Time'].apply(lambda x: x[-8:])
#-----------------


# fix dates
#-----------------

for timestr in ['Start Time', 'End Time']:
    for i in range(len(passengers_df)):
        if int(passengers_df[timestr].iloc[i][0:2]) > 23:
            passengers_df.drop(i)


min_ticks = np.arange(24*60+1)


def strtimediff(dt_start,dt_end):
    d = int(dt_end[0:2])*60+int(dt_end[3:5])+int(dt_end[6:8])/60 - (int(dt_start[0:2])*60+int(dt_start[3:5])+int(dt_start[6:8])/60)
    if (int(dt_start[0:2])>0 and int(dt_end[0:2])==0):
        d+=24*60
    return d

try:
    t_diff = [strtimediff(passengers_df['Start Time'].iloc[i],passengers_df['End Time'].iloc[i]) for i in range(len(passengers_df))]
except:
    print(passengers_df['End Time'])
#-----------------

# check if any passengers are in multiple times
# neg dates
#-----------------
passengers_df['Waiting Time'] = t_diff

for i in range(len(passengers_df)):
    if passengers_df['Waiting Time'].iloc[i]<0:
        input(passengers_df.iloc[i])


# print(passengers_df[passengers_df.duplicated()])
# print(passengers_df[passengers_df.duplicated(['Passenger #'])])

passengers_df.drop_duplicates(keep='first')
#-----------------


# passenger statistics
#-----------------
p_mean = passengers_df['Waiting Time'].mean()
p_std = passengers_df['Waiting Time'].std()

print(f'Waiting time - mean:{p_mean}, std:{p_std}')
no_max = passengers_df.drop(passengers_df['Waiting Time'].idxmax())

p_mean = no_max['Waiting Time'].mean()
p_std = no_max['Waiting Time'].std()

print(f'Waiting time without max - mean:{p_mean}, std:{p_std}')
#-------------


# >15 min wait
#-----------------
passengers_15 = passengers_df[passengers_df['Waiting Time']>15]

passengers_15['start time minutes'] = passengers_15['Start Time'].apply(lambda x: int(x[0:2])*60 + int(x[3:5]) + int(x[7:])/60) 


print(f'Percentage of passengers waiting longer than 15 minutes: {len(passengers_15)/len(passengers_df)}')

sns.histplot(data=passengers_df,x='Waiting Time')
plt.title('Waiting time in minutes')
plt.show()

sns.histplot(data = passengers_15, x = 'Percentage Time', y ='Waiting Time')
plt.show()
#-----------------


# total number of people at any time
#-----------------
passengers_df['Start Percentage Time'] = passengers_df['Start Time'].apply(lambda x: (int(x[0:2])*60+int(x[3:5]))/(24*60))
passengers_df['End Percentage Time'] = passengers_df['End Time'].apply(lambda x: (int(x[0:2])*60+int(x[3:5])+int(x[6:8])/60)/(24*60))

passengers_df['Start in minutes'] = passengers_df['Start Time'].apply(lambda x: (int(x[0:2])*60+int(x[3:5])))
passengers_df['End in minutes'] = passengers_df['End Time'].apply(lambda x: (int(x[0:2])*60+int(x[3:5])+int(x[6:8])/60))



# sum of people waiting

# def in_range(c,a,b):
#     """
#     check if c is in range of a,b
#     """
#     return (c>=a and c<b)

# chosen_passengers_mins = []
# total_passengers_waiting_mins = []
# for t in min_ticks:
#     a = passengers_df.apply(lambda x: in_range(t,x['Start in minutes'], x['End in minutes']),axis=1).values.sum()
#     c_p = passengers_df[passengers_df.apply(lambda x: in_range(t,x['Start in minutes'], x['End in minutes']),axis=1).values]
#     chosen_passengers_mins.extend(c_p['Passenger #'])
#     total_passengers_waiting_mins.append(a)


def get_mins(s_t,e_t):
    if e_t<s_t:
        s_t-=24*60-1
    m = list(range(s_t,int(e_t+1)))
    return m

a = passengers_df.apply(lambda x: get_mins(x['Start in minutes'], x['End in minutes']),axis=1)
all_min_occurence = a.sum()

occ_per_min = []
for t in min_ticks:
    occ_per_min.append(all_min_occurence.count(t))


print(f'Max number of passengers at a given time: {max(occ_per_min)}')

tot_df['Total Passengers Waiting'] = occ_per_min
m_df = pd.melt(tot_df, value_vars=['Bags', 'Total Passengers Waiting'], var_name='Item', value_name='Total', ignore_index=False)


timeplot_from_percent(df = m_df, y_p = 'Total',title='Comparison between number of bags and passengers waiting',hue='Item', legend=True)
#-----------------

