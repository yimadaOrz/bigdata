#!/usr/bin/env python
# coding: utf-8

from pyspark import SparkContext
from pyspark.sql.session import SparkSession
import csv
import numpy as np
import time 
import sys

def convert_house(h):
    if '-' in h:
        t = h.split('-')
        assert len(t) == 2
        house = int(t[0]) + int(t[1])/10**8
        flag = int(t[1])
    else:
        house = float(h)
        flag = int(house)
    return (house, flag)

def processCenterline(pid, records):
    # Skip the header
    if pid==0:
        next(records)
    reader = csv.reader(records)
    
    for row in reader:
        try:
            pysicalID = int(row[0])
            street_label = row[10].lower()
            full_street = row[28].lower()
            boro = int(row[13])
            l_low, _ = convert_house(row[2])
            l_high, _ = convert_house(row[3])
            r_low, _ = convert_house(row[4])
            r_high, _ = convert_house(row[5])
        except:
            (pysicalID, street_label, full_street, boro, l_low, l_high, r_low, r_high) = (int(row[0]), '', '', -1, -1.0, -1.0, -1.0, -1.0)
        if street_label == full_street:
            yield (pysicalID, street_label, boro, l_low, l_high, 1)
            yield (pysicalID, street_label, boro, r_low, r_high, 0)
        else:
            yield (pysicalID, street_label, boro, l_low, l_high, 1)
            yield (pysicalID, street_label, boro, r_low, r_high, 0)
            yield (pysicalID, full_street, boro, l_low, l_high, 1)
            yield (pysicalID, full_street, boro, r_low, r_high, 0)


def processViolation(pid, records):
    county2idx = {'MAN':1,'MH':1,'MN':1,'NEWY':1,'NEW Y':1,'NY':1,\
                  'BRONX':2,'BX':2,'PBX':2,\
                  'BK':3,'K':3,'KING':3,'KINGS':3,\
                  'Q':4,'QN':4,'QNS':4,'QU':4,'QUEEN':4,\
                  'R':5,'RICHMOND':5}
    # Skip the header
    if pid==0:
        next(records)
    reader = csv.reader(records)
    for row in reader:
        try:
            year = int(row[4][-4:])
            if year not in range(2015,2020):
                raise ValueError
            house_str = row[23]
            house, flag = convert_house(house_str)
            street = row[24].lower()
            if row[21] in county2idx.keys():
                boro = county2idx[row[21]]
            else:
                raise ValueError
            is_left = flag%2
        except:
            continue
        yield (year, house, street, boro, is_left)


def processformat(records):
    for r in records:
        if r[0][1]==2015:
            yield (r[0][0], (r[1], 0, 0, 0, 0))
        elif r[0][1]==2016:
            yield (r[0][0], (0, r[1], 0, 0, 0))
        elif r[0][1]==2017:
            yield (r[0][0], (0, 0, r[1], 0, 0))
        elif r[0][1]==2018:
            yield (r[0][0], (0, 0, 0, r[1], 0))
        elif r[0][1]==2019:
            yield (r[0][0], (0, 0, 0, 0, r[1]))
        else:
            yield (r[0][0], (0, 0, 0, 0, 0))


def compute_ols(y, x=list(range(2015,2020))):
    x, y = np.array(x), np.array(y)
    # number of observations/points 
    n = np.size(x)
    # mean of x and y vector 
    m_x, m_y = np.mean(x), np.mean(y)
    # calculating cross-deviation and deviation about x 
    SS_xy = np.sum(y*x) - n*m_y*m_x 
    SS_xx = np.sum(x*x) - n*m_x*m_x
    # calculating regression coefficients 
    coef = SS_xy / SS_xx
    return coef


def process(i):
    if ',' in str(i):
        return "\"{}\"".format(i)
    else:
        return str(i)


def to_csv(rdd):
    li = map(process, rdd)
    return ','.join(li)

if __name__ == "__main__":
    start_time = time.time()
    output = sys.argv[1]
    
    sc = SparkContext()
    spark = SparkSession(sc)

    centerline = sc.textFile('hdfs:///tmp/bdm/nyc_cscl.csv') \
    .mapPartitionsWithIndex(preprocessing)
    
    violations = sc.textFile('hdfs:///tmp/bdm/nyc_parking_violation/') \
    .mapPartitionsWithIndex(dataprocessing)
    
    vio = spark.createDataFrame(violations, ('year', 'street', 'boro', 'house_number', 'is_left'))
    cline = spark.createDataFrame(centerline, ('pysicalID', 'street', 'boro', 'low', 'high', 'is_left'))
    condition = [vio.boro == cline.boro, 
             vio.street == cline.street,
             vio.is_left == cline.is_left, 
             (vio.house_number >= cline.low) & (vio.house_number <= cline.high)]
    df = cline.join(vio, condition, how='left').groupBy([cline.pysicalID, vio.year]).count()
    
    df.rdd.map(lambda x: ((x[0], x[1]), x[2])) \
            .mapPartitions(processformat) \
            .reduceByKey(lambda x,y: (x[0]+y[0], x[1]+y[1], x[2]+y[2], x[3]+y[3], x[4]+y[4])) \
            .sortByKey() \
            .mapValues(lambda y: y + (compute_ols(y=list(y)),)) \
           .map(lambda x: ((x[0],)+x[1]))\
           .map(to_csv)\
           .saveAsTextFile(output)
    
    print('total running time : {} seconds'.format(time.time()-start_time))
