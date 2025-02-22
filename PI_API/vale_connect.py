# -*- coding: utf-8 -*-

import requests
import urllib3
import pandas
import numpy

from datetime import datetime

## Date Time Utilities
class ValeTime():

    ## Constructor
    def __init__(self):
        super(ValeTime,self).__init__()

    ## Time string to Timestamp string
    # @param string A date time string like '2024-01-01 00:00:00'
    def str2timestamp(self,str_in):
        dt_object = datetime.strptime(str_in,'%Y-%m-%d %H:%M:%S')
        return dt_object.strftime('%Y-%m-%dT%H:%M:%SZ')

## PIWebAPI request and Data Loading
class ValeConnect():

    ## Constructor
    # @param string Server IP or URL
    # @param string Base hierarchy path
    def __init__(self,server_path,base_path):
        super(ValeConnect, self).__init__()

        self.SERV_PATH = f'https://{server_path}/piwebapi/'
        self.BASE_PATH = base_path

        # disable SSL warnings
        urllib3.disable_warnings()

        # other class objects
        self.dt = ValeTime()

    ## generic GET request
    # @param string URL to send GET
    # @param string Key from JSON to retrieve value
    def api_get_request(self,get_url,key=''):
        resp = requests.get(get_url,verify=False)
        resp_json = resp.json()

        if len(key) == 0:
            return resp_json
        else:
            return resp_json[key]

    ## Get Web Id hash from PI Server on specified Tag ID
    # @param string Tag ID
    # @return Web Id string
    def get_webid_point(self,tag_path):
        url = f'{self.SERV_PATH}/points?path=\\\\{self.BASE_PATH}\{tag_path}'
        return self.api_get_request(url,'WebId')

    ## Get JSON record from PI Server on specified Web Id
    # @param string Wed Id
    # @param int Data amount
    # @return dict Raw server's response in JSON
    def get_stream_rec_json(self,web_id,cnt=200):
        int_cnt = int(cnt)
        if int_cnt >= 2000:
            str_cnt = '2000'
        else:
            str_cnt = str(int_cnt)

        url = f'{self.SERV_PATH}/streams/{web_id}/recorded?maxCount={str_cnt}'
        return self.api_get_request(url)

    ## Get Value record from PI Server on specified Web Id
    # @param string Wed Id
    # @param int Data amount
    # @return numpy One-dimensional Array of values
    def get_stream_rec_value(self,web_id,cnt=200):
        int_cnt = int(cnt)
        if int_cnt >= 2000:
            str_cnt = '2000'
        else:
            str_cnt = str(int_cnt)

        url = f'{self.SERV_PATH}/streams/{web_id}/recorded?maxCount={str_cnt}&selectedFields=Items.Value'
        val_list = self.api_get_request(url,'Items')
        val_np = numpy.zeros((len(val_list)))

        j = 0
        for i in val_list:
            val_np[j] = i.get('Value')
            j = j + 1

        return val_np

    ## Get TimeStamp Value record from PI Server on specified Web Id
    # @param string Wed Id
    # @param int Data amount
    # @return dict Two-dimensional of timestamps (string) and values (float)
    def get_stream_rec_valuetime_dict(self,web_id,cnt=200):
        int_cnt = int(cnt)
        if int_cnt >= 2000:
            str_cnt = '2000'
        else:
            str_cnt = str(int_cnt)

        url = f'{self.SERV_PATH}/streams/{web_id}/recorded?maxCount={str_cnt}&selectedFields=Items.Timestamp;Items.Value'
        val_list = self.api_get_request(url,'Items')
        val_dict = {}

        j = 0
        for i in val_list:
            val_new = {i.get('Timestamp'):i.get('Value')} # DO NOT SWAP
            val_dict.update(val_new)
            j = j + 1

        return val_dict

    ## Get TimeStamp/Value record from PI Server on specified Web Id
    # @param string Wed Id
    # @param int Data amount
    # @return pandas Two-dimensional of timestamps (string) and values (float)
    def get_stream_rec_valuetime_pd(self,web_id,cnt=200):
        int_cnt = int(cnt)
        if int_cnt >= 2000:
            str_cnt = '2000'
        else:
            str_cnt = str(int_cnt)

        url = f'{self.SERV_PATH}/streams/{web_id}/interpolated?maxCount={str_cnt}&interval=1m&selectedFields=Items.Timestamp;Items.Value' # recorded
        val_list = self.api_get_request(url,'Items')
        val_dict = {}

        j = 0
        for i in val_list:
            # note the Z is parsing directive to %z as it is identifier to UTC timezone
            val_new = {i.get('Timestamp'):i.get('Value')} # DO NOT SWAP
            val_dict.update(val_new)
            j = j + 1

        val_df = pandas.DataFrame(list(val_dict.items()), columns=['Timestamps','Values'])
        return val_df

    ## Get TimeStamp/Value record from PI Server on specified Web Id
    # @param string Wed Id
    # @param list Start and End of time span string like ['2024-01-01 00:00:00','2024-01-01 10:00:00']
    # @return pandas Two-dimensional of timestamps (string) and values (float)
    def get_stream_rec_valuetimespan_pd(self,web_id,time_span):
        start_dt = self.dt.str2timestamp(time_span[0])
        end_dt = self.dt.str2timestamp(time_span[1])

        url = f'{self.SERV_PATH}/streams/{web_id}/interpolated?startTime={start_dt}&endTime={end_dt}&interval=1m&selectedFields=Items.Timestamp;Items.Value'
        #print(url)
        val_list = self.api_get_request(url,'Items')
        val_dict = {}

        j = 0
        for i in val_list:
            # note the Z is parsing directive to %z as it is identifier to UTC timezone
            val_new = {i.get('Timestamp'):i.get('Value')} # DO NOT SWAP
            val_dict.update(val_new)
            j = j + 1

        val_df = pandas.DataFrame(list(val_dict.items()), columns=['Timestamps','Values'])
        return val_df
