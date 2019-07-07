def getPnext(pattern):
    """
    复杂版回溯法求前后缀得到next数组
    """
    plength = len(pattern)
    pnext = [0] * plength
    if plength == 1:
        return [0]
    else:
        pnext[1] = 0
        for i in range(2, plength):
            # pattern[i] 后缀 pattern[pnext[i-1]]前缀
            if pattern[i] == pattern[pnext[i - 1]]:
                pnext[i] = pnext[i - 1] + 1
            else:
                # 回溯
                tmpNext = pnext[i - 1]
                while tmpNext > 0:
                    if pattern[i] == pattern[pnext[tmpNext]]:
                        pnext[i] = pnext[tmpNext] + 1
                        break
                    else:
                        tmpNext = pnext[tmpNext]
                if tmpNext <= 0:
                    pnext[i] = 0

        return [-1] + pnext[:plength-1]


# # 思路清晰版 + 小优化
# def getPnext2(pattern):
#     i = -1 # 指向前缀
#     j = 0 # 指向后缀
#     plen = len(pattern)
#     pnext = [-1] * plen
#     while(j < plen - 1):
#         if i == -1 or pattern[i] == pattern[j]:
#             i += 1
#             j += 1
#             if pattern[i] != pattern[j]:
#                 pnext[j] = i
#             else:
#                 pnext[j] = pnext[i]
#         else:
#             i = pnext[i]
#     return pnext
#
# print(getPnext2('abcabc'))

def kmpMatch(sequence, pattern):
    pnext = getPnext(pattern)
    i = 0  # 指向sequence
    seqlen = len(sequence)
    j = 0  # 指向pattern
    plen = len(pattern)
    locList = []
    while i < seqlen:
        if j == -1 or sequence[i] == pattern[j]:
            i += 1
            j += 1
            if j == plen:
                # 匹配成功
                locList.append(i-j)
                j = 0
        else:
            j = pnext[j]

    return locList

print(kmpMatch('ABCDABCAC','AB'))

# 参考文献:https://www.cnblogs.com/ZuoAndFutureGirl/p/9028287.html