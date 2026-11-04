import { test, expect } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';

const root=fileURLToPath(new URL('../../',import.meta.url));
const python=root+(process.platform==='win32'?'.venv/Scripts/python.exe':'.venv/bin/python');

test('generated video + real ORT + durable outbox reaches UI/report then privacy deletion',async({page},info)=>{
  test.setTimeout(90000);
  const prepared=JSON.parse(execFileSync(python,['scripts/probe_synthetic_workflow.py','prepare'],{cwd:root,encoding:'utf8',timeout:60000,env:{...process.env,PYTHONPYCACHEPREFIX:root+'.cache/pycache'}}));
  const receipt=JSON.parse(readFileSync(root+prepared.private_receipt,'utf8'));
  const errors:string[]=[];
  page.on('pageerror',e=>errors.push(e.message));
  page.on('console',e=>{if(e.type()==='error')errors.push(e.text());});
  page.on('requestfailed',r=>{if(!r.failure()?.errorText.includes('ERR_ABORTED'))errors.push(r.failure()?.errorText||'request failed');});
  page.on('response',r=>{if(r.url().includes('/api/')&&r.status()>=400)errors.push(String(r.status()));});
  try {
    await page.goto('/');
    await page.getByLabel('기관 코드').fill(receipt.identity.org_code);
    await page.getByLabel('이메일 또는 로그인 ID').fill(receipt.identity.login);
    await page.getByLabel('비밀번호',{exact:true}).fill(receipt.identity.password);
    await page.getByRole('button',{name:'로그인',exact:true}).click();
    await expect(page.getByRole('heading',{name:'점검 개요',exact:true})).toBeVisible();
    await page.getByLabel('자료 출처 범위').selectOption('synthetic');
    await expect(page.locator('.row-target')).toHaveCount(2);
    await expect(page.locator('.row-target').first()).toContainText('합성');
    await page.locator('.row-target').first().click();
    const dialog=page.getByRole('dialog');
    await expect(dialog).toBeVisible();
    const image=dialog.locator('.evidence-image img');
    await expect(image).toBeVisible();
    expect(await image.evaluate((e:HTMLImageElement)=>e.complete&&e.naturalWidth===64&&e.naturalHeight===48)).toBeTruthy();
    await expect(dialog).toContainText('SYNTHETIC-GRAPH-CONTRACT-NOT-TRAINED');
    await dialog.getByRole('button',{name:'상세 닫기'}).click();
    await page.getByRole('button',{name:'보고서',exact:true}).click();
    await page.getByLabel('보고서 자료 출처').selectOption('synthetic');
    await page.getByRole('button',{name:'보고서 생성',exact:true}).click();
    await expect(page.locator('.job-card')).toBeVisible();
    for(let i=0;i<10&&!(await page.getByRole('button',{name:'다운로드',exact:true}).count());i++){
      await page.getByRole('button',{name:'보고서 상태 새로고침'}).click();
      await page.waitForTimeout(200);
    }
    const downloadEvent=page.waitForEvent('download');
    await page.getByRole('button',{name:'다운로드',exact:true}).click();
    const download=await downloadEvent;
    expect(download.suggestedFilename()).toContain('.csv');
    await page.setViewportSize({width:390,height:844});
    await page.getByRole('button',{name:'메뉴 열기'}).click();
    await page.getByRole('button',{name:'점검 목록',exact:true}).click();
    await expect(page.getByRole('heading',{name:'점검 목록',exact:true})).toBeVisible();
    await expect.poll(()=>page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
    expect(errors).toEqual([]);
    await info.attach('synthetic-workflow-contract',{body:JSON.stringify({aggregate:prepared.aggregate,browser_errors:errors}),contentType:'application/json'});
  } finally {
    const cleaned=JSON.parse(execFileSync(python,['scripts/probe_synthetic_workflow.py','cleanup','--receipt',prepared.private_receipt],{cwd:root,encoding:'utf8',timeout:30000,env:{...process.env,PYTHONPYCACHEPREFIX:root+'.cache/pycache'}}));
    expect(cleaned.privacy_delete_job_succeeded).toBe(true);
    expect(cleaned.prior_report_invalidated).toBe(true);
    expect(cleaned.server_tombstone_blocks_reupload).toBe(true);
    expect(cleaned.local_revocation_blocks_requeue).toBe(true);
    await info.attach('synthetic-workflow-deletion',{body:JSON.stringify(cleaned),contentType:'application/json'});
  }
});
