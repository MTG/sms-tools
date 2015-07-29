import os, re, json
from collections import namedtuple

try: #python 3
    from urllib.request import urlopen, FancyUrlOpener, Request
    from urllib.parse import urlparse, urlencode, quote
    from urllib.error import HTTPError
except ImportError: #python 2.7
    from urlparse import urlparse
    from urllib import urlencode, FancyURLopener, quote
    from urllib2 import HTTPError, urlopen, Request
    
class URIS():    
    HOST = 'www.freesound.org'
    BASE =  'https://'+HOST+'/apiv2'
    TEXT_SEARCH = '/search/text/'
    CONTENT_SEARCH= '/search/content/'
    COMBINED_SEARCH = '/sounds/search/combined/'
    SOUND = '/sounds/<sound_id>/'
    SOUND_ANALYSIS = '/sounds/<sound_id>/analysis/'
    SIMILAR_SOUNDS = '/sounds/<sound_id>/similar/'
    COMMENTS = '/sounds/<sound_id>/comments/'
    DOWNLOAD = '/sounds/<sound_id>/download/'
    UPLOAD = '/sounds/upload/'
    DESCRIBE = '/sounds/<sound_id>/describe/'
    PENDING = '/sounds/pending_uploads/'
    BOOKMARK = '/sounds/<sound_id>/bookmark/'
    RATE = '/sounds/<sound_id>/rate/'
    COMMENT = '/sounds/<sound_id>/comment/'
    AUTHORIZE = '/oauth2/authorize/'
    LOGOUT = '/api-auth/logout/'
    LOGOUT_AUTHORIZE = '/oauth2/logout_and_authorize/'
    ME = '/me/'
    USER = '/users/<username>/'
    USER_SOUNDS = '/users/<username>/sounds/'
    USER_PACKS = '/users/<username>/packs/'
    USER_BOOKMARK_CATEGORIES = '/users/<username>/bookmark_categories/'
    USER_BOOKMARK_CATEGORY_SOUNDS = '/users/<username>/bookmark_categories/<category_id>/sounds/'
    PACK = '/packs/<pack_id>/'
    PACK_SOUNDS = '/packs/<pack_id>/sounds/'
    PACK_DOWNLOAD = '/packs/<pack_id>/download/'   

    
    @classmethod
    def uri(cls, uri, *args):
        for a in args:
            uri = re.sub('<[\w_]+>', quote(str(a)), uri, 1)
        return cls.BASE+uri    

class FreesoundClient():

    client_secret = ""
    client_id = ""
    token = ""
    header =""
    
    def get_sound(self, sound_id):
        uri = URIS.uri(URIS.SOUND,sound_id)        
        return FSRequest.request(uri, {}, self, Sound)

    def text_search(self, **params):
        uri = URIS.uri(URIS.TEXT_SEARCH)
        return FSRequest.request(uri, params, self, Pager)

    def content_based_search(self, **params):
        uri = URIS.uri(URIS.CONTENT_SEARCH)
        return FSRequest.request(uri, params, self, Pager)
        
    def combined_search(self, **params):
        uri = URIS.uri(URIS.COMBINED_SEARCH)
        return FSRequest.request(uri,params,self,CombinedSearchPager)
    
    def get_user(self,username):
        uri = URIS.uri(URIS.USER, username)
        return FSRequest.request(uri,{},self,User)

    def get_pack(self,pack_id):
        uri = URIS.uri(URIS.PACK, pack_id)
        return FSRequest.request(uri,{},self,Pack)
    
    
    
    def set_token(self, token, auth_type="token"):
        self.token = token#TODO        
        self.header = 'Bearer '+token if auth_type=='oauth' else 'Token '+token    
        

class FreesoundObject:
    def __init__(self,json_dict, client):
        self.client=client
        def replace_dashes(d):
            for k, v in d.items():
                if "-" in k:
                    d[k.replace("-","_")] = d[k]
                    del d[k]
                if isinstance(v, dict):replace_dashes(v)
        
        replace_dashes(json_dict)
        self.__dict__.update(json_dict)
        for k, v in json_dict.items():
            if isinstance(v, dict):
                self.__dict__[k] = FreesoundObject(v, client)

class FreesoundException(Exception):
    def __init__(self, http_code, detail):        
        self.code = http_code
        self.detail = detail
    def __str__(self):
        return '<FreesoundException: code=%s, detail="%s">' % \
                (self.code,  self.detail)
        
class Retriever(FancyURLopener):
    def http_error_default(self, url, fp, errcode, errmsg, headers):
        resp = fp.read()
        try:
            error = json.loads(resp)
            raise FreesoundException(errcode,resp.detail)
        except:
            raise Exception(resp)
            
class FSRequest:
    @classmethod
    def request(cls, uri, params={}, client=None, wrapper=FreesoundObject, method='GET',data=False):
        p = params if params else {}
        url = '%s?%s' % (uri, urlencode(p)) if params else uri
        d = urllib.urlencode(data) if data else None
        headers = {'Authorization':client.header}  
        req = Request(url,d,headers)
        try:
            f = urlopen(req)
        except HTTPError, e:
            resp = e.read()
            if e.code >= 200 and e.code < 300:
                return resp
            else:
                raise FreesoundException(e.code,json.loads(resp))
        resp = f.read()
        f.close()
        result = None
        try:
            result = json.loads(resp)
        except:
            raise FreesoundException(0,"Couldn't parse response")
        if wrapper:
            return wrapper(result,client)
        return result

    @classmethod
    def retrieve(cls, url, client,path):
        r = Retriever()
        r.addheader('Authorization', client.header)        
        return r.retrieve(url, path)

class Pager(FreesoundObject):
    def __getitem__(self, key):
        return Sound(self.results[key],self.client)

    def next_page(self):
        return FSRequest.request(self.next, {}, self.client, Pager)

    def previous_page(self):
        return FSRequest.request(self.previous, {}, self.client, Pager)

class CombinedSearchPager(FreesoundObject):
    def __getitem__(self, key):
        return Sound(self.results[key], None)

    def more(self):
        return FSRequest.request(self.more, {}, self.client, CombinedSearchPager)

class Sound(FreesoundObject):
        
    def retrieve(self, directory, name=False):
        path = os.path.join(directory, name if name else self.name)
        uri = URIS.uri(URIS.DOWNLOAD, self.id)
        return FSRequest.retrieve(uri, self.client,path)
    
    def retrieve_preview(self, directory, name=False):
        path = os.path.join(directory, name if name else str(self.previews.preview_lq_mp3.split("/")[-1]))
        return FSRequest.retrieve(self.previews.preview_lq_mp3, self.client,path)

    def get_analysis(self, descriptors=None):
        uri = URIS.uri(URIS.SOUND_ANALYSIS,self.id)
        params = {}
        if descriptors:
            params['descriptors']=descriptors
        return FSRequest.request(uri, params,self.client,FreesoundObject)

    def get_similar(self):
        uri = URIS.uri(URIS.SIMILAR_SOUNDS,self.id)
        return FSRequest.request(uri, {},self.client, Pager)

    def get_comments(self):
        uri = URIS.uri(URIS.COMMENTS,self.id)
        return FSRequest.request(uri, {}, self.client, Pager)

    def __repr__(self):
        return '<Sound: id="%s", name="%s">' % \
                (self.id, self.name)

class User(FreesoundObject):

    def get_sounds(self):
        uri = URIS.uri(URIS.USER_SOUNDS,self.username)
        return FSRequest.request(uri, {}, self.client, Pager)    
    
    def get_packs(self):
        uri = URIS.uri(URIS.USER_PACKS,self.username)
        return FSRequest.request(uri, {}, self.client, Pager)    

    def get_bookmark_categories(self):
        uri = URIS.uri(URIS.USER_BOOKMARK_CATEGORIES,self.username)
        return FSRequest.request(uri, {}, self.client, Pager)    

    def get_bookmark_category_sounds(self): 
        uri = URIS.uri(URIS.USER_BOOKMARK_CATEGORY_SOUNDS,self.username)
        return FSRequest.request(uri, {}, self.client, Pager)    

    def __repr__(self): return '<User: "%s">' % ( self.username)

class Pack(FreesoundObject):

    def get_sounds(self):
        uri = URIS.uri(URIS.PACK_SOUNDS,self.id)
        return FSRequest.request(uri, {}, self.client, Pager)
    
    def __repr__(self):
        return '<Pack:  name="%s">' % \
                ( self.get('name','n.a.'))
