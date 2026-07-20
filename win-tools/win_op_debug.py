import sys, torch
print("PY torch", torch.__version__, "hip", torch.version.hip, "cuda?", torch.cuda.is_available(), flush=True)
def run(name, fn, expected):
    try:
        r = fn()
        torch.cuda.synchronize()
        got = r.cpu()
        ok = torch.allclose(got, expected.cpu())
        print(f"{name}: {'PASS' if ok else 'WRONG-VALUE'}  got={got.flatten().tolist()[:8]}  exp={expected.flatten().tolist()[:8]}", flush=True)
    except Exception as e:
        print(f"{name}: EXCEPTION {type(e).__name__} :: {str(e)[:400]}", flush=True)
d="cuda"
m1=torch.tensor([[1.,2.,3.],[4.,5.,6.]],device=d)
m2=torch.tensor([[7.,8.,9.,10.],[11.,12.,13.,14.],[15.,16.,17.,18.]],device=d)
run("mm_CONTROL", lambda: torch.mm(m1,m2), torch.tensor([[74.,80.,86.,92.],[173.,188.,203.,218.]]))
e1=torch.tensor([[7.,8.,9.],[10.,11.,12.]],device=d)
run("elementwise_mul", lambda: torch.tensor([[1.,2.,3.],[4.,5.,6.]],device=d)*e1, torch.tensor([[7.,16.,27.],[40.,55.,72.]]))
run("transpose", lambda: torch.t(torch.tensor([[1.,2.,3.],[4.,5.,6.]],device=d)).contiguous(), torch.tensor([[1.,4.],[2.,5.],[3.,6.]]))
run("dot", lambda: torch.dot(torch.tensor([1.,2.,3.],device=d),torch.tensor([4.,5.,6.],device=d)), torch.tensor(32.))
run("mv", lambda: torch.mv(m1,torch.tensor([7.,8.,9.],device=d)), torch.tensor([50.,122.]))
run("matmul", lambda: torch.matmul(m1,m2), torch.tensor([[74.,80.,86.,92.],[173.,188.,203.,218.]]))
sys.stdout.flush()
