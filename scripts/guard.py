"""Local process-group timeout and +256 MiB system swap watchdog."""
import argparse,json,os,re,signal,subprocess,time,ctypes
from pathlib import Path
class Usage(ctypes.Structure):
 _fields_=[('uuid',ctypes.c_uint8*16)]+[(k,ctypes.c_uint64) for k in ('user_time system_time pkg_idle_wkups interrupt_wkups pageins wired_size resident_size phys_footprint proc_start_abstime proc_exit_abstime child_user_time child_system_time child_pkg_idle_wkups child_interrupt_wkups child_pageins child_elapsed_abstime diskio_bytesread diskio_byteswritten').split()]
def swap():
 s=subprocess.check_output(['/usr/sbin/sysctl','vm.swapusage'],text=True)
 m=re.search(r'used\s*=\s*([0-9.]+)([KMGT])',s)
 if not m:raise ValueError('Cannot measure system swap')
 return float(m[1])*{'K':1/1024,'M':1,'G':1024,'T':1024**2}[m[2]]
def stop(p):
 if p.poll() is None:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=5)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True,type=Path);ap.add_argument('--timeout',type=float,default=300);ap.add_argument('command',nargs=argparse.REMAINDER);a=ap.parse_args()
 out=a.output;out.mkdir(parents=True,exist_ok=False);baseline=swap();start=time.monotonic();peak=0;growth=0;p=None
 receipt={'baseline_swap_mib':baseline,'swap_limit_mib':256,'timeout_seconds':a.timeout,'command':a.command,'status':'running'}
 lib=ctypes.CDLL('/usr/lib/libproc.dylib',use_errno=True)
 def interrupted(signum,frame):raise KeyboardInterrupt(f'Signal {signum}')
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
 try:
  with (out/'stdout.json').open('w') as stdout,(out/'stderr.log').open('w') as stderr:
   cmd=a.command[1:] if a.command[:1]==['--'] else a.command
   p=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,start_new_session=True)
   while p.poll() is None:
    growth=max(growth,swap()-baseline);u=Usage()
    if lib.proc_pid_rusage(p.pid,2,ctypes.byref(u))==0:peak=max(peak,u.phys_footprint)
    if (out/'cancel.request').exists():receipt['status']='cancelled';break
    if growth>256:receipt['status']='stopped_swap';break
    if time.monotonic()-start>a.timeout:receipt['status']='stopped_timeout';break
    time.sleep(.2)
   if receipt['status']=='running':receipt['status']='pass' if p.returncode==0 else 'failed'
 except BaseException:
  receipt['status']='interrupted_or_exception'
  raise
 finally:
  if p:stop(p)
  receipt.update(elapsed_seconds=time.monotonic()-start,peak_process_footprint_bytes=peak,positive_system_swap_growth_mib=growth,exit_code=p.returncode if p else None)
  (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt));return 0 if receipt['status']=='pass' else 1
if __name__=='__main__':raise SystemExit(main())
