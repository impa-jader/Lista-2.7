# Lista 7 - Prog 2 - Jader Duarte
# Questão 1
def find_judge(n,trust):
    pessoas={p+1 for p in range(n)}
    for i in trust:
        pessoas.discard(i[0]) # como todo mundo confiam no juiz, todos os não juizes confiam em alguem. Portanto o unico que não confia em ninguem é o juiz.  
    if pessoas:
        return pessoas.pop()
    else:
        return -1
    
if __name__=="__main__":
    t = [ [ 1 , 2 ] , [ 1 , 3 ] , [ 2 , 3 ] ]
    n = 3
    print(find_judge(n, t))  # Saída: 3
