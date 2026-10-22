/* تشخیص خودکار لوگو و عکس اعضا روی بنر (الگوی ثابت بنرهای انجمن) - ورودی: {data:RGBA, width, height} */
function detectBanner(img){
  const W=img.width,H=img.height,f=Math.max(1,Math.round(W/556)),w=Math.floor(W/f),h=Math.floor(H/f);
  const gold=new Uint8Array(w*h);
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){
    const i=((y*f+(f>>1))*W+(x*f+(f>>1)))*4,r=img.data[i],g=img.data[i+1],b=img.data[i+2];
    if(r>140&&r>=g&&g>b&&r-b>70&&g>85)gold[y*w+x]=1;
  }
  const lab=new Int32Array(w*h),comps=[];let n=0;const st=[];
  for(let s=0;s<w*h;s++){
    if(!gold[s]||lab[s])continue;n++;lab[s]=n;st.push(s);
    let x0=w,x1=0,y0=h,y1=0,cnt=0;
    while(st.length){const p=st.pop(),px=p%w,py=(p/w)|0;cnt++;
      if(px<x0)x0=px;if(px>x1)x1=px;if(py<y0)y0=py;if(py>y1)y1=py;
      for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){const nx=px+dx,ny=py+dy;
        if(nx<0||ny<0||nx>=w||ny>=h)continue;const q=ny*w+nx;if(gold[q]&&!lab[q]){lab[q]=n;st.push(q)}}}
    comps.push({x0,x1,y0,y1,cnt});
  }
  let logo=null;const mem=[];
  for(const c of comps){
    const bw=c.x1-c.x0+1,bh=c.y1-c.y0+1,sz=(bw+bh)/2,asp=Math.abs(bw-bh)/Math.max(bw,bh);
    if(asp>0.12||c.cnt<sz*3)continue;
    const cx=(c.x0+c.x1+1)/2*f,cy=(c.y0+c.y1+1)/2*f,rel=sz/w;
    if(rel>0.2&&rel<0.34&&cy<H*0.3){if(!logo||sz>logo.sz)logo={x:cx,y:cy,r:sz/2*f,sz}}
    else if(rel>0.10&&rel<0.19&&cx>W*0.7)mem.push({x:cx,y:cy,r:sz/2*f});
  }
  mem.sort((a,b)=>a.y-b.y);
  const L=logo;const mm=L?mem.filter(m=>m.y>L.y+L.r*1.1):mem;
  const out=[];for(const m of mm){if(!out.length||m.y-out[out.length-1].y>m.r)out.push(m)}
  return{logo:logo&&{x:logo.x,y:logo.y,r:logo.r},members:out};
}
if(typeof module!=='undefined')module.exports={detectBanner};
