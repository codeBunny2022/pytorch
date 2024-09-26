n=int(input())
while n>0:
    s=list(input().split(" "))
    res=[]
    for i in s:
        if i.isupper():
            res.append(i)
        else:
            i=list(i)
            i[0]=i[0].upper()
            i=''.join(i)
            res.append(i)
    print(" ".join(res))
    n-=1