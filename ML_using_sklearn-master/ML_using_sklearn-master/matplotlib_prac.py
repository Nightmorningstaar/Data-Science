from matplotlib import pyplot as plt
from matplotlib import style

# ploting line graph
# style.use('ggplot')
#
# x1 = [5,8,10]
# y1 = [12,16,6]
# x2 = [6,9,11]
# y2 = [6,15,7]

# plt.plot(x1,y1,'g',label = 'line-one',linewidth = 5)
# plt.plot(x2,y2,'c',label = 'line-two',linewidth = 5)
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('info')
# plt.legend()
# plt.grid(True,color = 'k')
# plt.show()

#bar graph
# plt.bar([1,3,5,7,9],[5,2,7,8,2],label = 'exaple one',color = 'c')
# plt.bar([2,4,6,8,10],[8,6,2,5,6],label = 'example two',color = 'g')
# plt.legend()
# plt.xlabel('x- axis')
# plt.ylabel('y-axis')
# plt.title('info')
# plt.grid(True,color = 'k')
# plt.show()
# Bar graph (catagorical)

# agr humko alag alag show krna h
# plt.bar([1,3,5,7,9],[5,2,7,8,2],label = 'example one')
# plt.show()
# plt.bar([2,4,6,8,10],[8,6,2,5,6],label = 'example two')
# plt.show()
# plt.bar([11,13,15,17,19],[18,16,12,15,16],label = 'example three')
# plt.show()

# Agr ek saath krna h to
# plt.bar([1,3,5,7,9],[5,2,7,8,2],label = 'example one')
# plt.bar([2,4,6,8,10],[8,6,2,5,6],label = 'example two')
# plt.bar([11,13,15,17,19],[18,16,12,15,16],label = 'example three')
# plt.show()

# histogram
# population_ages = [2,55,62,23,34,4,5,6,57,56,78,12,
#                    45,23 ,56, 87, 12, 23, 45, 45, 78, 90 ]
# gdp = [0,10,20,30,40,50,60,70,80,90,100,120]
# plt.hist(population_ages,gdp,histtype = 'bar',rwidth = 0.5 )
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('india_gdp')
# plt.legend()
# plt.show()

#scatter plot
# x = [1,2,3,4,5,6,6,7]
# y = [5,3,1,2,5,6,8,5]
# plt.scatter(x,y,label = 'scatter',color = 'k')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('scatter-plot(')
# plt.legend()
# plt.show()

#area graph
days = [1,2,3,4,5]

sleep = [7,8,6,11,7]
eating = [2,3,4,3,2]
working = [7,8,7,2,2]
play = [8,5,7,8,13]

plt.plot(color ='m',label = 'sleep',linewidth = 5)
plt.plot(color ='c',label = 'eating',linewidth = 5)
plt.plot(color ='r',label = 'working',linewidth = 5)
plt.plot(color ='k',label = 'play',linewidth = 5)

plt.stackplot(days,sleep,eating,working,play,colors = ['m','c','r','k'])

plt.xlabel('x')
plt.ylabel('y')
plt.title('stack-plot')
plt.legend()
plt.show()

#pie chart
slices = [7,2,2,13]
activities = ['sleep','eat','work','play']
cols = ['c','m','r','b']
plt.pie(slices,labels = activities,
        colors = cols,
        startangle = 90,
        shadow = True,
        explode = (0,0.2,0,0),
        autopct = '% 1.1f%%')
plt.title("pie-chart")
plt.show()