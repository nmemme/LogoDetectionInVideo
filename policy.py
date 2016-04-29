import base64

import hmac, hashlib

AWS_SECRET_ACCESS_KEY = r"NP5iuTvRf6z2GuNKbuWl+HrHvz6UvNoPZTS2721g"

policy_document ='''
    
    {
    
    "expiration": "2020-01-01T00:00:00Z",
    
    "conditions": [
    
    {"bucket": "trackingobject"},
    
    ["starts-with", "$key", ""],
    
    {"acl": "public-read"},
    
    ["starts-with", "$Content-Type", ""],
    
    ["content-length-range", 0, 1048576000]
    
    ]
    
    }
'''
policy = base64.b64encode(policy_document)

signature = base64.b64encode(hmac.new(AWS_SECRET_ACCESS_KEY, policy, hashlib.sha1).digest())

print policy
print 'hi'
print signature