import os
import sys
import csv
import time
import calendar

import numpy as np

KSIZE = 20

def foo0():
  with open('Sheet1.csv') as csvfile:
    data = csv.reader(csvfile)
    flag = False
    for x in data:
      if flag:
        return x
      print(x)
      print([x for x in range(0, 15)])
      flag = True

def foo1():
  print(float(''.join('31,423,521.00'.split(','))))

def foo2():
  t = time.strptime('2017/3/24', '%Y/%m/%d')
  print(t)
  print(calendar.timegm(t) / 60 / 60 / 24)


def to_float(str):
  if str == '':
    return 0
  else:
    return float(''.join(str.split(',')))

def check_OK(l):
  for i in range(0, 15):
    if l[i] == '':
      return False
  return True

def foo3(x):
  if check_OK(x):
    ret = [
      int(x[0].split('.')[0]), #0
      time.strptime(x[2], '%Y-%m-%d').tm_year, #1
      time.strptime(x[2], '%Y-%m-%d').tm_yday, #2
      time.strptime(x[2], '%Y-%m-%d').tm_mon, #3
      time.strptime(x[2], '%Y-%m-%d').tm_mday, #4
      time.strptime(x[2], '%Y-%m-%d').tm_wday, #5
      to_float(x[3]), #6
      to_float(x[4]), #7 TO PRED
      to_float(x[5]), #8 TO PRED
      to_float(x[6]), #9
      to_float(x[7]), #10
      to_float(x[8]), #11
      to_float(x[9]), #12
      to_float(x[10]), #13 ####
      to_float(x[11]) / 1000., #14
      to_float(x[12]) / 1000., #15
      to_float(x[13]) / 1000., #16
      to_float(x[14]) / 1000., #17 ###
      int(calendar.timegm(time.strptime(x[2], '%Y-%m-%d')) / 60 / 60 / 24), #18
    ]
  else:
    ret = None
  return ret


def foo4():
  ret = []
  with open('Sheet1.csv') as csvfile:
    data = csv.reader(csvfile)
    flag = False
    for x in data:
      if flag:
        tmp = foo3(x)
        if tmp:
          ret.append(tmp)
      else:
        flag = True
  return ret

def foo5(k=KSIZE):
  v = {}
  for x in U_data:
    t = v.get(x[0], (0, 0))
    t = (t[0] + 1, t[1] + x[13] * x[13] * x[17])
    v[x[0]] = t
  w = []
  for x in v:
    s = v[x]
    w.append(s[1] / s[0])
  w.sort()
  b = w[-1-k]
  ret = []
  for x in v:
    s = v[x]
    if s[1] / s[0] > b:
      ret .append(x)
  return ret

def foo6(date=17240, l = 20):
  id = ID
  ids = {}
  i = 0
  for x in id:
    ids[x] = i
    i += 1
  data = []
  for x in U_data:
    if 0 <= x[-1] - date and x[-1] - date < l and x[0] in ids:
      data.append(x)
  ret = np.ndarray(shape=[l, KSIZE, 19], dtype=np.float32)
  for i in range(0, l):
    for x in data:
      if x[-1] == date + i:
        ret[i][ids[x[0]]] = np.array(x, dtype=np.float32)
  return ret

U_data = foo4()
ID = foo5()
