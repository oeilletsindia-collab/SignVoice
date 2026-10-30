"""Run against an already-started local app; capture actual UI, not mockups."""
import argparse,json
from pathlib import Path
from playwright.sync_api import sync_playwright

def check(url,output):
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1440,'height':1100})
        page.goto(url,wait_until='networkidle')
        page.get_by_role('tab',name='Upload Video',exact=True).wait_for()
        page.screenshot(path=str(out/'upload.png'),full_page=True)
        page.get_by_role('tab',name='Webcam Recognition',exact=True).click()
        page.get_by_text('This translates a completed recording',exact=False).wait_for()
        page.screenshot(path=str(out/'webcam.png'),full_page=True)
        page.get_by_role('tab',name='Model Information',exact=True).click()
        page.get_by_text('no trained checkpoint installed',exact=False).wait_for()
        page.screenshot(path=str(out/'model-information.png'),full_page=True)
        browser.close()
    print(json.dumps(dict(status='passed',checks=['app HTTP response','upload tab','webcam recording tab','honest missing-checkpoint state'],
                         limitations=['No real camera used','No trained translation tested'],screenshots=str(out))))
if __name__=='__main__':
    a=argparse.ArgumentParser(); a.add_argument('--url',default='http://127.0.0.1:7860'); a.add_argument('--output',default='assets/screenshots')
    x=a.parse_args(); check(x.url,x.output)
