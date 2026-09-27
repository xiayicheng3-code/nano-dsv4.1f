"""Stage packing, shifted SFT labels, deterministic inputs, and checkpoint contracts."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prepare_posttrain_corpus as data
import prepare_trace_corpus as adapters
import posttrain_input as inputs
import run_posttrain as runner
from nano_dsv41f.trace_corpus import ASSISTANT_TOKEN_ID, assistant_sft_loss_mask, render_case_v41
from nano_dsv41f.training import causal_lm_loss
from nano_dsv41f.pretrain_recipe import pretrain_recipe, recipe_manifest
from nano_dsv41f.chat_protocol import EOS_TOKEN_ID, SPECIAL_TOKENS


def case(source='swe_success'):
    return {'metadata':{'source':source},'messages':[], 'thinking_mode':'chat'}


def test_full_prefix_16k_preserves_history_removed_from_8k():
    ids = [0,ASSISTANT_TOKEN_ID] + [40]*1000 + [EOS_TOKEN_ID] + [41]*7600 + [ASSISTANT_TOKEN_ID,42,EOS_TOKEN_ID]
    short = data.prefix_view(ids,case(),8192)
    long = data.prefix_view(ids,case(),16384)
    assert len(short.tokens) == 1003
    assert len(long.tokens) == len(ids) > 8192
    np.testing.assert_array_equal(long.tokens,ids)
    assert not long.sft_loss_mask[1003:8603].any()


def test_pivot_never_drops_target_or_supervises_previous_assistant():
    ids = [0,ASSISTANT_TOKEN_ID,40,EOS_TOKEN_ID] + [41]*9000 + [ASSISTANT_TOKEN_ID,42,EOS_TOKEN_ID]
    assert data.prefix_view(ids,case('nemotron_conversational_pivot'),8192) is None
    view = data.prefix_view(ids,case('nemotron_conversational_pivot'),16384)
    assert view.sft_loss_mask.sum() == 2
    assert not view.sft_loss_mask[:-2].any()


def test_long_single_answer_is_not_cut_in_the_middle():
    ids = [0,ASSISTANT_TOKEN_ID] + [40]*9000 + [EOS_TOKEN_ID]
    assert data.prefix_view(ids,case('openr1_math'),8192) is None
    assert len(data.prefix_view(ids,case('openr1_math'),16384).tokens) == len(ids)


def test_sft_masks_targets_after_shift_and_keeps_prompt_context():
    ids = jnp.array([[0,1,2,3,4,5]])
    seg = jnp.array([[0,0,0,0,1,1]])
    targets = jnp.array([[False,False,True,True,True,False]])
    logits = jnp.zeros((1,6,6))
    loss,count = causal_lm_loss(logits,ids,seg,target_mask=targets)
    assert int(count) == 2  # target 4 crosses segments, prompt target 1 has no loss
    poisoned = logits.at[0,0,:].set(100*jnp.arange(6)).at[0,3,:].set(-100*jnp.arange(6))
    other,n = causal_lm_loss(poisoned,ids,seg,target_mask=targets)
    assert float(other) == float(loss) and int(n) == 2
    grads = jax.grad(lambda x:causal_lm_loss(x,ids,seg,target_mask=targets)[0])(logits)
    assert not np.asarray(grads[0,0]).any()
    assert np.asarray(grads[0,1]).any()  # first supervised target sees previous context token
    assert np.asarray(grads[0,2]).any()


def test_observations_can_be_preserved_without_changing_legacy_default():
    from nano_dsv41f.trace_corpus import truncate_text
    text='x'*7000
    assert truncate_text(text,None) == text
    assert len(truncate_text(text,100)) < 7000
    source=replace(next(s for s in adapters.AGENT_SOURCES if s.key=='openthoughts_execution'),
                   preserve_history=True,max_observation_chars=None)
    raw=[{'role':'system','content':'Use terminal'}, {'role':'user','content':'Task'}]
    for i in range(6):
        raw += [{'role':'assistant','content':json.dumps({'analysis':'inspect','commands':[{'keystrokes':'ls\n'}]})},
                {'role':'user','content':text+str(i)}]
    raw += [{'role':'assistant','content':json.dumps({'analysis':'done','commands':[],'task_complete':True})}]
    adapted=adapters.adapt_openthoughts(source,{'conversations':raw},0)
    assert adapted is not None
    assert len([m for m in adapted['messages'] if m['role']=='tool']) == 6
    assert adapted['messages'][3]['content'] == text+'0'
    assert '[truncated' not in render_case_v41(adapted)


def test_writer_reader_real_roundtrip_and_resume_cursor(tmp_path):
    writer=data.Writer(tmp_path,8192,1701,3)
    for i in range(13):
        ids=np.array([0,ASSISTANT_TOKEN_ID]+[40+i]*8000+[EOS_TOKEN_ID],np.uint16)
        writer.add(data.prefix_view(ids,case(),8192))
    manifest=writer.finish()
    shards=[(tmp_path/s['file'],s) for s in manifest['shards']]
    pool=inputs.Pool(shards,seed=9,length=8192)
    one=[pool.batch(i) for i in range(3)]
    resumed=inputs.Pool(shards,seed=9,length=8192)
    for k in one[2]:np.testing.assert_array_equal(one[2][k],resumed.batch(2)[k])
    count=inputs.batch_counts(one[0],8192,sft=True)
    _, actual=causal_lm_loss(jnp.zeros((4,8192,60)),jnp.array(one[0]['input_ids']),
        jnp.array(one[0]['segment_ids']),token_mask=jnp.array(one[0]['token_mask']),
        target_mask=jnp.array(one[0]['sft_loss_mask']))
    assert count['lm_tokens'] == int(actual)
    with pytest.raises(StopIteration):pool.batch(3)


def test_token_fair_sampler_is_deterministic_after_restore():
    mix={'document':.8,'reasoning':.05,'agent':.15}
    counts=dict.fromkeys(mix,0)
    for i in range(1000):
        name=inputs.choose_pool(counts,mix)
        counts[name]+=15000 + (i*17)%10000
    total=sum(counts.values())
    for name,w in mix.items(): assert abs(counts[name]/total-w)<.002
    assert inputs.choose_pool(counts,mix) == inputs.choose_pool(json.loads(json.dumps(counts)),mix)


def test_sft_recipe_changes_positions_and_mask_but_not_parameter_shapes():
    config,train,native=pretrain_recipe(profile='narrow48',cp=2,dp=4,benchmark_total_steps=100000)
    base=json.loads(json.dumps(recipe_manifest(config,train,native)))
    mid,mt,mn=runner.stage_recipe(base,'midtrain',20_000_000,2.6e-5)
    sft,st,sn=runner.stage_recipe(base,'sft',20_000_000,2.6e-5)
    assert mt == train and mid == config
    assert st.seq_len == 16384 and st.learning_rate == 2.6e-5
    assert sft.attention.rope.original_seq_len == 8192 and sft.attention.rope.rope_factor == 2
    assert sft.indexer_training.apply_candidate_mask
    assert not mid.indexer_training.apply_candidate_mask
    assert (sft.d_model,sft.n_layers,sft.n_experts)==(mid.d_model,mid.n_layers,mid.n_experts)


def test_base_requires_complete_matching_tokenizer_and_remaining_budget():
    state={'stage':'pretrain','real_tokens':2_400_001_000,'identity':{
        'base_tokens':2_400_000_000,'total_tokens':3_000_000_000,
        'corpus':{'tokenizer_sha256':'abc'}}}
    corpus={'identity':{'tokenizer_sha256':'abc'},'midtrain_target_tokens':600_000_000}
    runner.validate_base(state,corpus)
    with pytest.raises(ValueError,match='Complete'):runner.validate_base({**state,'real_tokens':10},corpus)
    with pytest.raises(ValueError,match='tokenizers'):runner.validate_base(state,{**corpus,'identity':{'tokenizer_sha256':'def'}})


def test_sft_budget_respects_smaller_pool_at_independent_ratio():
    manifest={'sources':[{'pool':'reasoning','views':{'sft':{'train':{'real_tokens':4_000_000}}}},
                         {'pool':'agent','views':{'sft':{'train':{'real_tokens':16_000_000}}}}],
              'pool_mix':{'sft':{'reasoning':1/3,'agent':2/3}}}
    budget=inputs.sft_budget(manifest)
    assert 11_000_000 < budget < 12_000_000


def test_notebooks_compile_and_include_distinct_entrypoints():
    import nbformat
    root=Path(__file__).resolve().parents[1]
    for name,script in [('prepare_midtrain8k_sft16k_cpu','prepare_posttrain_corpus.py'),
                        ('midtrain8k_sft16k_tpu','run_posttrain.py')]:
        nb=nbformat.read(root/'notebooks'/f'nano_dsv41f_{name}.ipynb',as_version=4)
        nbformat.validate(nb)
        for cell in nb.cells:
            if cell.cell_type=='code':compile(cell.source,name,'exec')
        assert script in '\n'.join(c.source for c in nb.cells)


def test_openthoughts_real_think_json_and_empty_wait_command():
    msg = adapters._terminal_action('<think>Wait for test output.</think>{"analysis":"waiting",'
        '"commands":[{"keystrokes":"","duration":10}]}',call_id='wait')
    assert msg['reasoning_content'].startswith('Wait for test output.')
    assert json.loads(msg['tool_calls'][0]['function']['arguments'])['commands'][0]['keystrokes']==''


def test_harmony_research_parser_merges_analysis_and_checks_answer():
    def msg(role,text,**kw):return {'role':role,'content':[{'type':'text','text':text}],**kw}
    raw=[msg('user','Find author'),msg('assistant','I should search.',channel='analysis'),
         msg('assistant','{"query":"author"}',channel='analysis',recipient='browser.search'),
         msg('tool','Ada wrote it.',name='browser.search'),
         msg('assistant','The evidence agrees.',channel='analysis'),
         msg('assistant','Explanation: source\nExact Answer: Ada\nConfidence: 99%',channel='final')]
    messages=adapters._harmony_context(raw,0,None)
    assert len(messages)==4
    assert messages[1]['tool_calls'][0]['function']['name']=='browser.search'
    assert messages[1]['reasoning_content']=='I should search.'
    source=replace(next(s for s in adapters.AGENT_SOURCES if s.key=='openresearcher'),
                   preserve_history=True,max_observation_chars=None)
    row={'messages':raw,'answer':r'\boxed{Ada}','status':'success','error':None}
    assert adapters.adapt_openresearcher(source,row,0) is not None
    assert adapters.adapt_openresearcher(source,{**row,'answer':r'\boxed{Bob}'},0) is None
    assert not adapters.research_reference_matches({'answer':'1.5'},[{'role':'assistant','content':'Exact Answer: 15'}])


def test_task_split_groups_pivot_contexts_and_research_seeds():
    first={'metadata':{'trajectory_id':42},'messages':[{'role':'user','content':'one'}]}
    second={'metadata':{'trajectory_id':42},'messages':[{'role':'user','content':'two'}]}
    assert data.task_key(first)==data.task_key(second)


def test_runner_transitions_and_resumes_with_real_checkpoints(tmp_path,monkeypatch):
    """Exercise both stages/control flow using real input files and CPU state trees."""
    import time
    from types import SimpleNamespace
    import nano_dsv41f as nano
    import nano_dsv41f.tpu as tpu
    import nano_dsv41f.tpu_native as native
    from pretrain_checkpoint import save_checkpoint,read_metadata,load_checkpoint
    from jax.sharding import SingleDeviceSharding
    root=tmp_path/'corpus';root.mkdir()
    (root/'tokenizer.json').write_text('{}')
    token_hash=hashlib.sha256(b'{}').hexdigest()
    def shard(directory,length):
        directory.mkdir(parents=True)
        ids=np.full((32,length),40,np.uint16);ids[:,0]=0;ids[:,1]=ASSISTANT_TOKEN_ID;ids[:,-1]=EOS_TOKEN_ID
        target=np.ones_like(ids,np.uint8);target[:,:2]=0
        file=directory/'shard.npz'
        np.savez(file,input_ids=ids,segment_ids=np.zeros_like(ids),token_mask=np.ones_like(ids,np.uint8),sft_loss_mask=target)
        return {'file':file.name,'rows':32,'bytes':file.stat().st_size,'sha256':hashlib.sha256(file.read_bytes()).hexdigest()}
    manifest={'format':inputs.FORMAT,'complete':True,'identity':{'tokenizer_sha256':token_hash},
        'midtrain_target_tokens':131072,'sources':[],
        'pool_mix':{'midtrain':{'document':.8,'reasoning':.05,'agent':.15},'sft':{'reasoning':1/3,'agent':2/3}}}
    d=shard(root/'documents/midtrain',8192)
    manifest['documents']={'path':'documents/midtrain','real_tokens':32*8192,'shards':[d]}
    for pool in ('reasoning','agent'):
        views={}
        for stage,length in [('midtrain',8192),('sft',16384)]:
            meta=shard(root/stage/pool/'train',length)
            views[stage]={'train':{'real_tokens':32*length,'rows':32,'shards':[meta]},'validation':{'shards':[],'rows':0}}
        manifest['sources'].append({'source':{'key':pool},'pool':pool,'views':views})
    (root/'posttrain_manifest.json').write_text(json.dumps(manifest))
    sharding=SingleDeviceSharding(jax.devices()[0])
    def trees():return {'w':jnp.ones((2,),jnp.bfloat16)},{'w':jnp.zeros((2,),jnp.float32)}
    params,opt=trees()
    c,t,n=pretrain_recipe(profile='narrow48',cp=2,dp=4,benchmark_total_steps=100000)
    base={'stage':'pretrain','completed_steps':100,'real_tokens':1000000,
          'identity':{'base_tokens':1000000,'total_tokens':1131072,'corpus':{'tokenizer_sha256':token_hash}},
          'recipe':json.loads(json.dumps(recipe_manifest(c,t,n)))}
    basepath=save_checkpoint(tmp_path/'base',(params,opt),base)
    monkeypatch.setattr(nano,'validate_v5e_runtime',lambda:None)
    monkeypatch.setattr(nano,'make_v5e_mesh',lambda:None)
    monkeypatch.setattr(nano,'init_model_sharded_mixed_precision',lambda *a,**kw:(trees()[0],None,None))
    resets=[]
    def init_opt(*a):resets.append(True);return trees()[1],None
    monkeypatch.setattr(nano,'init_optimizer_state_sharded',init_opt)
    monkeypatch.setattr(nano,'put_training_batch',lambda ids,seg,mask,*a:tuple(jnp.array(v) for v in (ids,seg,mask)))
    monkeypatch.setattr(tpu,'batch_named_sharding',lambda *a:sharding)
    monkeypatch.setattr(tpu,'named_shardings',lambda *a:{'w':sharding})
    monkeypatch.setattr(native,'NativeCompiledStep',lambda fn,*a:fn)
    stages=[]
    def compile_step(p,o,spec,config,train,mesh,**kw):
        stages.append((train.seq_len,kw['assistant_only']))
        def step(p,o,ids,seg,number,mask,targets=None):
            valid=mask[:,1:]&mask[:,:-1]&(seg[:,1:]==seg[:,:-1])
            if targets is not None:valid &= targets[:,1:]
            m={'loss':jnp.array(1.),'lm_loss':jnp.array(1.),'indexer_loss':jnp.array(0.),
               'lm_tokens':valid.sum(),'expert_dropped':jnp.array(0),
               'moe_mosaic_layers':jnp.array(config.n_layers),'learning_rate':jnp.array(1e-5)}
            return {'w':p['w']+jnp.array(.125,jnp.bfloat16)},{'w':o['w']+1},m
        return step
    monkeypatch.setattr(nano,'compile_pretrain_step',compile_step)
    monkeypatch.setattr(nano,'compile_diagnostics',lambda fn,*args:(fn,{'control_plane_test':True}))
    def args(out,resume=None,max_steps=0):
        return SimpleNamespace(corpus=root,output=out,pretrained=basepath if not resume else None,
            resume=resume,sft_tokens=196608,sft_lr=2.6e-5,seed=1701,
            deadline_unix=time.time()+7200,checkpoint_every=2,eval_every=100,
            eval_batches=1,log_every=100,max_steps=max_steps)
    runner.run(args(tmp_path/'first',max_steps=2))
    first=json.loads((tmp_path/'first/summary.json').read_text())
    assert first['status']=='paused' and first['stage']=='midtrain'
    runner.run(args(tmp_path/'resumed',Path(first['checkpoint'])))
    finished=json.loads((tmp_path/'resumed/summary.json').read_text())
    assert finished['status']=='completed' and finished['stage']=='sft'
    _,meta=read_metadata(finished['checkpoint'])
    assert meta['metadata']['stage_complete'] and meta['metadata']['stage_steps']==3
    assert meta['metadata']['pool_tokens']=={'agent':131072,'reasoning':65536}
    # Four midtrain updates then reset optimizer; final SFT moments count only 3 updates.
    restored,_=load_checkpoint(finished['checkpoint'],trees())
    np.testing.assert_array_equal(np.asarray(restored[1]['w']),[3.,3.])
    assert (8192,False) in stages and (16384,True) in stages
    # Run uninterrupted and compare the actual final parameter/optimizer arrays.
    runner.run(args(tmp_path/'uninterrupted'))
    final=json.loads((tmp_path/'uninterrupted/summary.json').read_text())
    whole,_=load_checkpoint(final['checkpoint'],trees())
    for a,b in zip(jax.tree.leaves(restored),jax.tree.leaves(whole)):
        np.testing.assert_array_equal(np.asarray(a),np.asarray(b))


def test_jitted_sft_compiler_passes_separate_mask_without_masking_context(monkeypatch):
    import nano_dsv41f.tpu as tpu
    from nano_dsv41f.config import ModelConfig,TrainConfig
    from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
    mesh=Mesh(np.array(jax.devices()[:1]),('tp',))
    sharding=NamedSharding(mesh,P())
    monkeypatch.setattr(tpu,'named_shardings',lambda *a:{'w':sharding})
    monkeypatch.setattr(tpu,'optimizer_state_named_shardings',lambda *a:{'w':sharding})
    monkeypatch.setattr(tpu,'batch_named_sharding',lambda *a:sharding)
    def step(p,opt,config,train,ids,*,token_mask,target_mask=None,**kw):
        count=token_mask.sum() if target_mask is None else target_mask.sum()
        return {'w':p['w']+count},{'w':opt['w']+1},{'visible':token_mask.sum(),'targets':count}
    monkeypatch.setattr(tpu,'pretrain_step',step)
    cfg,train=ModelConfig(),TrainConfig()
    p={'w':jax.device_put(jnp.array(0.),sharding)};opt={'w':jax.device_put(jnp.array(0.),sharding)}
    fn=tpu.compile_pretrain_step(p,opt,None,cfg,train,mesh,include_indexer=False,assistant_only=True)
    ids=jnp.ones((1,8),jnp.int32);seg=jnp.zeros_like(ids);visible=jnp.ones_like(ids,bool)
    targets=jnp.array([[False,False,False,False,False,True,True,True]])
    _,_,metrics=fn(p,opt,ids,seg,jnp.array(0),visible,targets)
    assert int(metrics['visible'])==8 and int(metrics['targets'])==3


def test_real_renderer_tokenizer_collection_writes_disjoint_stage_records(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import gzip
    from tokenizers import Tokenizer,models,pre_tokenizers,AddedToken
    vocab={token:i for i,token in enumerate(SPECIAL_TOKENS)}
    vocab['[UNK]']=len(vocab)
    tok=Tokenizer(models.WordLevel(vocab,unk_token='[UNK]'))
    tok.pre_tokenizer=pre_tokenizers.Whitespace()
    tok.add_special_tokens([AddedToken(x,special=True) for x in SPECIAL_TOKENS])
    source=next(s for s in adapters.REASONING_SOURCES if s.key=='chimera_science')
    rows=[{'question':f'Question {i}','solution':'Reason '* (i+1),'answer':'42',
           'subject':'Physics','correctness':True} for i in range(100)]
    monkeypatch.setattr(data,'source_stream',lambda *a:iter(rows))
    args=SimpleNamespace(output=tmp_path,midtrain_tokens=1_000_000,
                         sft_reasoning_tokens=1_000_000,sft_agent_tokens=1_000_000,
                         sft_long_token_fraction=0.0,headroom=1.1,seed=1701,
                         shard_rows=2,tokenize_batch_size=16)
    result=data.build_trace_source(source,args,tok)
    assert result['collection']['canonical_records']==100
    assert result['views']['midtrain']['train']['real_tokens']>0
    assert result['views']['sft']['train']['supervised_tokens']>0
    with gzip.open(tmp_path/result['canonical'],'rt') as f:
        saved=[json.loads(x) for x in f]
    assert len(saved)==100
    assert all(1<=c['reasoning_effort']<=100 for c in saved)
    assert len({c['metadata']['task_sha256'] for c in saved}) == 100
    assert result['selection']['cross_stage_task_overlap'] == 0
    assert sum(result['views'][stage][split].get('records', 0)
               for stage in ('midtrain','sft') for split in ('train','validation')) == 100
    for stage,length in [('midtrain',8192),('sft',16384)]:
        for split in ('train','validation'):
            for shard in result['views'][stage][split]['shards']:
                with np.load(tmp_path/stage/source.key/split/shard['file']) as a:
                    assert a['input_ids'].shape[1]==length
                    assert not (a['sft_loss_mask'].astype(bool)&~a['token_mask'].astype(bool)).any()


def selection_fixture(midtrain_tokens=2_000, sft_reasoning_tokens=200, long_fraction=.5):
    source = replace(next(s for s in adapters.REASONING_SOURCES if s.key == 'chimera_science'), weight=1.0)
    from types import SimpleNamespace
    args = SimpleNamespace(midtrain_tokens=midtrain_tokens,
        sft_reasoning_tokens=sft_reasoning_tokens, sft_agent_tokens=200,
        sft_long_token_fraction=long_fraction, headroom=1.0)
    return source, data.StageSelection(source, args)


def trace_of_length(length):
    return data.TokenizedTrace(np.zeros(length, np.uint16), np.ones(length, np.uint8),
                               'chimera_science', 0, 0, {})


def task_with_parity(source, parity, prefix='task'):
    for i in range(10_000):
        task = f'{prefix}-{i}'
        digest = hashlib.sha256((source.key + '\n' + task).encode()).hexdigest()
        if int(digest[8:16], 16) % 2 == parity:
            return task
    raise AssertionError('could not find a stable task hash')


def test_sft_selection_continues_after_midtrain_fills_and_short_cannot_fill_long():
    source, selection = selection_fixture()
    mid_task = task_with_parity(source, 1)
    assert selection.select(mid_task, {'midtrain': trace_of_length(100),
                                       'sft': trace_of_length(100)})[0] == 'midtrain'
    sft_task = task_with_parity(source, 0, 'sft')
    assert selection.select(sft_task, {'midtrain': trace_of_length(100),
                                       'sft': trace_of_length(100)})[0] == 'sft'
    assert selection.tokens['midtrain'] == selection.targets['midtrain']
    assert selection.tokens['sft_short'] == selection.targets['sft_short']
    short_late = task_with_parity(source, 0, 'short-late')
    assert selection.select(short_late, {'midtrain': trace_of_length(100),
                                         'sft': trace_of_length(100)}) is None
    long_only = 'late-long-answer'
    selected = selection.select(long_only, {'midtrain': None, 'sft': trace_of_length(10_000)})
    assert selected[0:3] == ('sft', 'train', 'sft_long')
    assert selection.done
    assert selection.audit()['cross_stage_task_overlap'] == 0


def test_variants_of_an_owned_task_cannot_cross_stages():
    source, selection = selection_fixture()
    task = task_with_parity(source, 1, 'owned')
    assert selection.select(task, {'midtrain': trace_of_length(100),
                                   'sft': trace_of_length(100)})[0] == 'midtrain'
    assert selection.select(task, {'midtrain': None, 'sft': trace_of_length(10_000)}) is None
    assert selection.audit()['cross_stage_task_overlap'] == 0


def test_sft_quotas_are_independent_of_midtrain_and_xlam_is_short_only():
    source, selection = selection_fixture(midtrain_tokens=20_000, sft_reasoning_tokens=400)
    assert selection.targets['midtrain'] == 1_000
    assert selection.targets['sft_short'] == 200
    assert selection.targets['sft_long'] == 200
    xlam = replace(next(s for s in adapters.AGENT_SOURCES if s.key == 'xlam_verified'), weight=1.0)
    from types import SimpleNamespace
    xsel = data.StageSelection(xlam, SimpleNamespace(midtrain_tokens=20_000,
        sft_reasoning_tokens=200, sft_agent_tokens=400, sft_long_token_fraction=.5, headroom=1.0))
    task = task_with_parity(xlam, 0, 'xlam')
    assert xsel.select(task, {'midtrain': trace_of_length(100), 'sft': trace_of_length(10_000)}) is None
    short = xsel.select('xlam-short', {'midtrain': None, 'sft': trace_of_length(100)})
    assert short[0:3] == ('sft', 'train', 'sft_short')
    assert xsel.targets['sft_long'] == 0


def test_old_shared_corpus_is_rejected(tmp_path):
    (tmp_path / 'posttrain_manifest.json').write_text(json.dumps({
        'format': 'nano-dsv41f-posttrain-v1', 'complete': True}))
    with pytest.raises(ValueError, match='old shared'):
        inputs.inspect(tmp_path)


def test_builder_reads_later_long_reasoning_after_midtrain_quota(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tokenizers import Tokenizer, models, pre_tokenizers, AddedToken
    vocab = {token: i for i, token in enumerate(SPECIAL_TOKENS)}
    vocab['[UNK]'] = len(vocab)
    tok = Tokenizer(models.WordLevel(vocab, unk_token='[UNK]'))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens([AddedToken(x, special=True) for x in SPECIAL_TOKENS])
    source = next(s for s in adapters.REASONING_SOURCES if s.key == 'chimera_science')
    question = next(f'first-{i}' for i in range(1000)
                    if int(hashlib.sha256((source.key + '\n' + f'first-{i}').encode()).hexdigest()[8:16], 16) % 2)
    rows = [
        {'question': question, 'solution': 'Reason ' * 40, 'answer': '42',
         'subject': 'Physics', 'correctness': True},
        {'question': 'later-long', 'solution': 'Reason ' * 10_000, 'answer': '42',
         'subject': 'Physics', 'correctness': True},
    ]
    monkeypatch.setattr(data, 'source_stream', lambda *a: iter(rows))
    args = SimpleNamespace(output=tmp_path, midtrain_tokens=2_000,
        sft_reasoning_tokens=200, sft_agent_tokens=200, sft_long_token_fraction=0.5,
        headroom=1.0, seed=1701, shard_rows=2, tokenize_batch_size=1)
    result = data.build_trace_source(source, args, tok)
    assert result['collection']['rows_seen'] == 2
    assert result['views']['midtrain']['train']['records'] == 1
    assert result['views']['sft']['train']['records'] == 1
    assert result['views']['sft']['train']['genuine_over_8k_records'] == 1
    assert result['selection']['cross_stage_task_overlap'] == 0
