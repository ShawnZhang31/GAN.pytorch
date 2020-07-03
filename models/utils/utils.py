# Copyright shawnzhang31. All Rights Reserved

def printProgressBar(iteration:int, 
                     total:int, 
                     prefix:str = '', 
                     suffix:str = '', 
                     decimals:int = 1, 
                     length:int = 100, 
                     fill:str = '#'):
    """
    输出进度条
    @params:
        iteration       - Required : 当前的迭代次数 (int)
        total           - Required : 总共要迭代的次数 (int)
        prefix          - Optional : 前缀字符串, 默认为空 (str)
        suffix          - Optional : 后缀字符串, 默认为空 (str)
        decimals        - Optional : 完成的百分比, 默认为1 (int)
        length          - Optional : 进度条的长度, 默认为100 (int)
        fill            - Optional : 进度条的填充字符， 默认为# (str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / total))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    if iteration == total:
        print()