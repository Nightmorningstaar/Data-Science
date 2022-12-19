# def bubble(l):
#     swap = 0
#     for i in range(len(l) - 1,0,-1):
#         for j in range(i):
#             if l[j] > l[j + 1]:
#                 l[j],l[j + 1] = l[j + 1],l[j]
#                 swap+=1
#             elif l[j] < l[j + 1]:
#                 continue
#     return swap

def selection_sort(l):
    swap = 0
    for i in range(len(l) - 1):
        minpos = i
        for j in range(i,len(l)):
            if l[minpos] > l[j]:
                minpos = j
        l[i],l[minpos] = l[minpos],l[i]
        swap += 1
    return swap

print(selection_sort([1 ,3, 5, 2, 4, 6, 7]))

