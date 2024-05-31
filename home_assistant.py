"""
For more information on the Home Assistant service calls:
    https://www.home-assistant.io/integrations/light/#service-lightturn_on
For more information on YeeLighting features:
    https://yeelight.readthedocs.io/en/stable/flow.html
Author: John Elliott
Date:   21/03/2024
"""
from requests       import post
from time           import sleep
from dotenv         import load_dotenv
from os             import environ


"""
Send the post request.

Parameters:
    url:     URL of the post request
    headers: Header of the post request with the auth data. 
    data:    Service call data being made.
"""
def call_service(url, headers, data):
    return post(url, headers=headers, json=data)


"""
For testing the lighting service calls.
"""
def test_light_services():
    url     = "http://homeassistant.local:8123/api/services/light/"
    headers = {"Authorization": "Bearer " + environ.get('HOME_ASSISTANT_LONG_TERM_KEY')}
    data    = {"entity_id":     "light."  + environ.get('LIGHTING_ID')}
    print()
    call_service(url + "turn_on", headers, data)
    sleep(5)
    call_service(url + "turn_off", headers, data)


"""
Creates the initial RESTFUL call. Can be set to mock this call and return nothing.

Parameters:
    state:       The desired state of the light, on or off. By default, this is set to on.
    brightness:  An integer representation of the light's brightness. Only recognized by light.turn_on call. By default, this is set to maximum.
    mock (bool): If True, the API call to Home Assistant will not be made. This is used for testing purposes. By default, this is set to False.
"""
def light_service(state="turn_on", brightness=255, mock=False):
    if not mock:
        url     = "http://homeassistant.local:8123/api/services/light/"
        headers = {"Authorization": "Bearer " + environ.get('HOME_ASSISTANT_LONG_TERM_KEY')}
        if state == "turn_on":
            data    = {"entity_id":     "light."  + environ.get('LIGHTING_ID'), "brightness": brightness}
        else:
            data    = {"entity_id":     "light."  + environ.get('LIGHTING_ID')}
        return call_service(url + state, headers, data)


"""
Loads the environmental variables.
"""
def load_envs():
    load_dotenv()


if __name__ == "__main__":
    load_envs()
    test_light_services()

