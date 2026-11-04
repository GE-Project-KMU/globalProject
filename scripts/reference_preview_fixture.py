"""Synthetic format/security fixture; NEVER a road photo or AI evidence.

Only reproduces the known pack metadata schema to exercise the untrusted file
parser. All pixels are generated here. No RTK photograph is read or distributed.
"""
import base64,io,json
from PIL import Image,ImageDraw

def main():
    image=Image.new('RGB',(160,64),(50,60,65));ImageDraw.Draw(image).text((5,8),'SYNTHETIC VALIDATOR',fill='white')
    stream=io.BytesIO();image.save(stream,format='JPEG')
    value={'schema_version':1,'pack_name':'SYNTHETIC FORMAT VALIDATOR - NOT RTK PHOTOS','source_name':'Road Traversing Knowledge (RTK), UFSC / Mendeley Data','attribution':'Thiago Rateke, Aldo von Wangenheim, Rafael Toledo · DOI 10.17632/hssswvmjwf.1','license_id':'CC-BY-4.0','modifications':'Synthetic parser fixture only','expires_at':'2027-02-01T16:51:19.893991+00:00','source_expires_at':'2027-02-01T16:51:19.893991+00:00','local_only':True,'human_privacy_reviewed':False,'inference_performed':False,'images':[{'title':'SYNTHETIC VALIDATOR '+str(i),'data_url':'data:image/jpeg;base64,'+base64.b64encode(stream.getvalue()).decode(),'width':160,'height':64} for i in range(3)]}
    print(json.dumps(value))
if __name__=='__main__':main()
