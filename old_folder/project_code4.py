import numpy as np
import csv
import matplotlib.pyplot as plt 
from numpy.polynomial.polynomial import polyfit
from scipy import optimize

with open("airline_delay.csv") as csvfile:
    reader = csv.DictReader(csvfile)
    data = [r for r in reader]

NUM = len(data)

#print NUM

totalyear=17
totalmonth = 12
delay_causes = 7

company_list =[]

selectcompnay_list = ['AA','UA','DL','AS']
selectcompnay_num = len(selectcompnay_list)

for n in range (0,NUM):
  s=data[n]['carrier']
  if not s in company_list:
    company_list.append(s)

#print(company_list)

company_num = len(company_list)

year_plot = np.array(range(2003, 2020))
month_plot = np.array([1,2,3,4,5,6,7,8,9,10,11,12])

delay_month = np.zeros((company_num, totalmonth, delay_causes+1))

delay_month2 = np.zeros((selectcompnay_num, totalmonth))
delay_month2_totfly = np.zeros((selectcompnay_num, totalmonth))


for n in range (0,NUM):
    company_index = company_list.index(data[n]['carrier'])
    month_index = int(data[n]['month'])-1
    if (data[n]['arr_cancelled']):
        delay_month[company_index][month_index][0] += float(data[n]['carrier_ct'])
        delay_month[company_index][month_index][1] += float(data[n]['weather_ct'])
        delay_month[company_index][month_index][2] += float(data[n]['nas_ct'])
        delay_month[company_index][month_index][3] += float(data[n]['security_ct'])
        delay_month[company_index][month_index][4] += float(data[n]['late_aircraft_ct'])
        delay_month[company_index][month_index][5] += float(data[n]['arr_cancelled'])
        delay_month[company_index][month_index][6] += float(data[n]['arr_diverted'])
        delay_month[company_index][month_index][7] += float(data[n]['arr_flights'])

    if ((data[n]['carrier'] in selectcompnay_list) and (data[n]['airport'] == 'ORD')):
        selectcompnay_index = selectcompnay_list.index(data[n]['carrier'])
        month_index = int(data[n]['month'])-1
        if (data[n]['arr_cancelled']):
            delay_month2[selectcompnay_index][month_index] += float(data[n]['carrier_ct'])
            delay_month2[selectcompnay_index][month_index] += float(data[n]['weather_ct'])
            delay_month2[selectcompnay_index][month_index] += float(data[n]['nas_ct'])
            delay_month2[selectcompnay_index][month_index] += float(data[n]['security_ct'])
            delay_month2[selectcompnay_index][month_index] += float(data[n]['late_aircraft_ct'])
            delay_month2[selectcompnay_index][month_index] += float(data[n]['arr_cancelled'])
            delay_month2[selectcompnay_index][month_index] += float(data[n]['arr_diverted'])
            delay_month2_totfly[selectcompnay_index][month_index] += float(data[n]['arr_flights'])

plot_list = [[] for x in range(delay_causes+1)]


for cause in range (0,delay_causes+1):
    for month in range (0, totalmonth):
        sum_company = 0
        for company in range (0,company_num):
          sum_company += delay_month[company][month][cause]
        plot_list[cause].append(sum_company)

for month in range (0,totalmonth):
    total_number_onemonth = plot_list[7][month]
    for cause in range (0,delay_causes):
        plot_list[cause][month] = plot_list[cause][month]* 100/total_number_onemonth


for C in range (selectcompnay_num):
    #sum_temp = 0  
    for M in range (0,totalmonth):
        delay_month2[C][M] = delay_month2[C][M]* 100/delay_month2_totfly[C][M]

#print(delay_month2)


def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / 4 / stddev)**2)


fig1 = plt.figure(figsize=(8,9))

ax1 = fig1.add_subplot(211)

ax1.plot(month_plot,plot_list[0],linewidth=3,color='blue')
ax1.plot(month_plot,plot_list[1],linewidth=3,color='violet')
ax1.plot(month_plot,plot_list[2],linewidth=3,color='black')
ax1.plot(month_plot,plot_list[3],linewidth=3,color='green')
ax1.plot(month_plot,plot_list[4],linewidth=3,color='red')
ax1.plot(month_plot,plot_list[5],linewidth=3,color='m')
ax1.plot(month_plot,plot_list[6],linewidth=3,color='coral')
ax1.xaxis.set_tick_params(labelsize=12,direction='in')
ax1.yaxis.set_tick_params(labelsize=12,direction='in')
ax1.legend(['carrier delay','weather delay','nas delay','security delay','late aircraft','canceled','diverted'], fontsize=11,handlelength=3,
    frameon=True,loc='upper center', bbox_to_anchor=(0.5, 1.23),fancybox=True, shadow=True, ncol=4)
ax1.set_ylabel('Delay Percentage (%)',fontsize=14)
ax1.set_xlim([1,12])
ax11 = ax1.twinx()
ax11.set_ylim(ax1.get_ylim())
ax11.get_yaxis().set_tick_params(direction='in',labelright='False')
plt.setp(ax1.get_xticklabels(),visible=False)
plt.title('Delay Flights as Functions of Month and Gaussian Fit',fontsize=14)


plt.subplots_adjust(hspace=0)

ax2 = fig1.add_subplot(212)

X0 = month_plot[3:9]
X2 = month_plot[3:9]
X4 = month_plot[3:9]

data0 = np.array(plot_list[0][3:9])
data2 = np.array(plot_list[2][3:9])
data4 = np.array(plot_list[4][3:9])

#print(data0+data2+data4)

data_ave = (data0  + data2  + data4)/3.0

#print(data_ave)

popt0, _ = optimize.curve_fit(gaussian, X0, data0)
popt2, _ = optimize.curve_fit(gaussian, X2, data2)
popt4, _ = optimize.curve_fit(gaussian, X4, data4)
popt_ave, _ = optimize.curve_fit(gaussian, X4, data_ave)

X_smooth = np.linspace(4, 9.5, 100)
ax2.plot(X_smooth,gaussian(X_smooth, *popt0),linewidth=3,color='blue')
ax2.plot(X_smooth,gaussian(X_smooth, *popt2),linewidth=3,color='black')
ax2.plot(X_smooth,gaussian(X_smooth, *popt4),linewidth=3,color='red')
ax2.plot(X_smooth,gaussian(X_smooth, *popt_ave),'r--',color='orange',linewidth=1.5)

ax2.plot(month_plot,plot_list[0],'o',color='blue')
ax2.plot(month_plot,plot_list[2],'o',color='black')
ax2.plot(month_plot,plot_list[4],'o',color='red')

ax2.set_xlabel('Month',fontsize=14)
ax2.xaxis.set_ticks(np.arange(1,13,1))
ax2.set_ylabel('Delay Percentage (%)',fontsize=14)
ax2.xaxis.set_tick_params(labelsize=12,direction='in')
ax2.yaxis.set_tick_params(labelsize=12,direction='in')
ax2.legend(['carrier delay','nas delay','late aircraft','Averaged'], fontsize=9,handlelength=4,frameon=True,loc='upper right')
ax2.set_xlim([1,12])
ax21 = ax2.twinx()
ax21.set_ylim(ax2.get_ylim())
ax21.get_yaxis().set_tick_params(direction='in',labelright='False')

fig1.subplots_adjust(left=0.08,right=0.97,bottom=0.05,top=0.9)

plt.savefig('delay_month_tot.pdf', format='pdf')


#------------------------------------------------------------------------------


fig3 = plt.figure(figsize=(8,8))

ax1 = fig3.add_subplot(211)


ax1.plot(month_plot,delay_month2[0],color='black')
ax1.plot(month_plot,delay_month2[1],color='violet')
ax1.plot(month_plot,delay_month2[2],color='blue')
ax1.plot(month_plot,delay_month2[3],color='green')
ax1.legend(['AA','UA','DL','AS'], fontsize=9,handlelength=4,frameon=True,loc='upper right')
ax1.xaxis.set_tick_params(labelsize=12,direction='in')
ax1.yaxis.set_tick_params(labelsize=12,direction='in')
ax1.xaxis.set_ticks(np.arange(1,13,1))
#ax1.set_xlabel('Month',fontsize=12)
ax1.set_ylabel('Delay Percentage (%)',fontsize=14)
ax1.set_xlim([1,12])
ax11 = ax1.twinx()
ax11.set_ylim(ax1.get_ylim())
ax11.get_yaxis().set_tick_params(direction='in',labelright='False')
plt.title('Delay Flights (Four Companies) at ORD Airport Sorted by Months and Gaussian Fitting',fontsize=14)
plt.setp(ax1.get_xticklabels(),visible=False)



plt.subplots_adjust(hspace=0)

ax2 = fig3.add_subplot(212)

X0 = month_plot[3:9]
X1 = month_plot[3:9]
X2 = month_plot[3:9]
X3 = month_plot[3:9]

#print(X0)



data0 = delay_month2[0][3:9]
data1 = delay_month2[1][3:9]
data2 = delay_month2[2][3:9]
data3 = delay_month2[3][3:9]

data_ave = (data0 + data1 + data2 + data3)/4.0

popt0, _ = optimize.curve_fit(gaussian, X0, data0)
popt1, _ = optimize.curve_fit(gaussian, X1, data1)
popt2, _ = optimize.curve_fit(gaussian, X2, data2)
popt3, _ = optimize.curve_fit(gaussian, X3, data3)
popt_ave, _ = optimize.curve_fit(gaussian, X3, data_ave)

X_smooth = np.linspace(4, 9.5, 100)
ax2.plot(X_smooth,gaussian(X_smooth, *popt0),color='black')
ax2.plot(X_smooth,gaussian(X_smooth, *popt1),color='violet')
ax2.plot(X_smooth,gaussian(X_smooth, *popt2),color='blue')
ax2.plot(X_smooth,gaussian(X_smooth, *popt3),color='green')
ax2.plot(X_smooth,gaussian(X_smooth, *popt_ave),'r--',color='orange',linewidth=5)

ax2.plot(month_plot,delay_month2[0],'o',color='black')
ax2.plot(month_plot,delay_month2[1],'o',color='violet')
ax2.plot(month_plot,delay_month2[2],'o',color='blue')
ax2.plot(month_plot,delay_month2[3],'o',color='green')
ax2.legend(['AA','UA','DL','AS','Averaged'], fontsize=9,handlelength=4,frameon=True,loc='upper right')
ax2.set_xlabel('Month',fontsize=14)
ax2.set_ylabel('Delay Percentage (%)',fontsize=14)
ax2.xaxis.set_tick_params(labelsize=12,direction='in')
ax2.yaxis.set_tick_params(labelsize=12,direction='in')
ax2.set_xlim([1,12])
ax2.xaxis.set_ticks(np.arange(1,13,1))
ax21 = ax2.twinx()
ax21.set_ylim(ax2.get_ylim())
ax21.get_yaxis().set_tick_params(direction='in',labelright='False')

fig3.subplots_adjust(left=0.08,right=0.97,bottom=0.08,top=0.92)


plt.savefig('delay_month_ORD.pdf', format='pdf')

