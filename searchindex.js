Search.setIndex({docnames:["gallery/attention-single-gpu","gallery/index","gallery/mlp-multi-gpu","gallery/quick-start","gallery/sg_execution_times","genindex","index","python_api/framework_dialect/deepspeed/engine","python_api/framework_dialect/deepspeed/index","python_api/framework_dialect/deepspeed/pipeline","python_api/framework_dialect/index","python_api/framework_dialect/registry","python_api/index","python_api/initialization","python_api/model_schedule/api","python_api/model_schedule/index","python_api/op/attention","python_api/op/cross_entropy","python_api/op/fused_bias","python_api/op/index","python_api/op/linear","python_api/op/mlp","python_api/pattern","python_api/pipeline","python_api/random","python_api/root","python_api/schedule","python_api/tracer","scripts/README","setup/index"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":5,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.intersphinx":1,"sphinx.ext.todo":2,"sphinx.ext.viewcode":1,sphinx:56},filenames:["gallery/attention-single-gpu.rst","gallery/index.rst","gallery/mlp-multi-gpu.rst","gallery/quick-start.rst","gallery/sg_execution_times.rst","genindex.rst","index.rst","python_api/framework_dialect/deepspeed/engine.rst","python_api/framework_dialect/deepspeed/index.rst","python_api/framework_dialect/deepspeed/pipeline.rst","python_api/framework_dialect/index.rst","python_api/framework_dialect/registry.rst","python_api/index.rst","python_api/initialization.rst","python_api/model_schedule/api.rst","python_api/model_schedule/index.rst","python_api/op/attention.rst","python_api/op/cross_entropy.rst","python_api/op/fused_bias.rst","python_api/op/index.rst","python_api/op/linear.rst","python_api/op/mlp.rst","python_api/pattern.rst","python_api/pipeline.rst","python_api/random.rst","python_api/root.rst","python_api/schedule.rst","python_api/tracer.rst","scripts/README.rst","setup/index.rst"],objects:{"":[[25,0,0,"-","slapo"]],"slapo.Database":[[25,2,1,"","commit"],[25,2,1,"","load"]],"slapo.ModulePattern":[[25,2,1,"","forward"]],"slapo.OrderedDict":[[25,2,1,"","clear"],[25,2,1,"","copy"],[25,2,1,"","fromkeys"],[25,2,1,"","items"],[25,2,1,"","keys"],[25,2,1,"","move_to_end"],[25,2,1,"","pop"],[25,2,1,"","popitem"],[25,2,1,"","setdefault"],[25,2,1,"","update"],[25,2,1,"","values"]],"slapo.Pattern":[[25,2,1,"","forward"]],"slapo.Space":[[25,2,1,"","cfg_dict_to_str"],[25,2,1,"","clone"],[25,2,1,"","create_symbol"],[25,2,1,"","log_space"],[25,2,1,"","next"],[25,2,1,"","reset"],[25,2,1,"","to_dict"]],"slapo.Symbol":[[25,2,1,"","add"],[25,2,1,"","fix_at"],[25,2,1,"","is_fixed"]],"slapo.framework_dialect":[[11,0,0,"-","registry"]],"slapo.framework_dialect.deepspeed":[[7,0,0,"-","engine"],[9,0,0,"-","pipeline"]],"slapo.framework_dialect.deepspeed.engine":[[7,4,1,"","init_ds_engine"]],"slapo.framework_dialect.deepspeed.pipeline":[[9,1,1,"","DeepSpeedPipeStageWrapper"],[9,1,1,"","WrappedTypeCode"],[9,4,1,"","analyze_tie_ranks"],[9,4,1,"","decode_metadata"],[9,4,1,"","deepspeed_pipe_engine"],[9,4,1,"","encode_metadata"],[9,4,1,"","flat_and_name_tensor_list"],[9,4,1,"","flatten"],[9,4,1,"","get_simple_nested_list_str"],[9,4,1,"","unflatten"]],"slapo.framework_dialect.deepspeed.pipeline.DeepSpeedPipeStageWrapper":[[9,2,1,"","forward"]],"slapo.framework_dialect.registry":[[11,4,1,"","get_all_dialects"],[11,4,1,"","get_dialect_cls"],[11,4,1,"","register_framework_dialect"]],"slapo.initialization":[[13,4,1,"","init_empty_weights"],[13,4,1,"","init_on_device"]],"slapo.model_schedule":[[14,0,0,"-","api"]],"slapo.model_schedule.api":[[14,4,1,"","apply_schedule"]],"slapo.op":[[16,0,0,"-","attention"],[17,0,0,"-","cross_entropy"],[18,0,0,"-","fused_bias"],[20,0,0,"-","linear"],[21,0,0,"-","mlp"]],"slapo.op.attention":[[16,1,1,"","FlashAttention"],[16,1,1,"","FlashAttentionOp"],[16,4,1,"","flash_attn_ref"],[16,4,1,"","get_xfoemers_attn_op_by_name"],[16,4,1,"","validate_sm_version"],[16,4,1,"","warning_once"],[16,4,1,"","xformers_ref"]],"slapo.op.attention.FlashAttention":[[16,2,1,"","forward"],[16,2,1,"","reshape_for_scores"]],"slapo.op.attention.FlashAttentionOp":[[16,2,1,"","forward"]],"slapo.op.cross_entropy":[[17,1,1,"","ParallelCrossEntropy"],[17,4,1,"","vocab_parallel_cross_entropy"]],"slapo.op.cross_entropy.ParallelCrossEntropy":[[17,2,1,"","forward"]],"slapo.op.fused_bias":[[18,1,1,"","BiasGeLUFunction"],[18,4,1,"","new_gelu"]],"slapo.op.fused_bias.BiasGeLUFunction":[[18,2,1,"","backward"],[18,2,1,"","forward"]],"slapo.op.linear":[[20,1,1,"","FusedQKV"],[20,1,1,"","LinearWithAct"],[20,1,1,"","LinearWithDropout"],[20,1,1,"","LinearWithSeparateBias"],[20,1,1,"","LinearWithSyncFunc"]],"slapo.op.linear.FusedQKV":[[20,2,1,"","forward"]],"slapo.op.linear.LinearWithAct":[[20,2,1,"","extra_repr"],[20,2,1,"","forward"]],"slapo.op.linear.LinearWithDropout":[[20,2,1,"","extra_repr"],[20,2,1,"","forward"]],"slapo.op.linear.LinearWithSeparateBias":[[20,2,1,"","forward"]],"slapo.op.linear.LinearWithSyncFunc":[[20,2,1,"","extra_repr"],[20,2,1,"","forward"]],"slapo.op.mlp":[[21,1,1,"","FusedMLP"]],"slapo.op.mlp.FusedMLP":[[21,2,1,"","forward"]],"slapo.partial":[[25,3,1,"","args"],[25,3,1,"","func"],[25,3,1,"","keywords"]],"slapo.pattern":[[22,1,1,"","ModulePattern"],[22,1,1,"","Pattern"]],"slapo.pattern.ModulePattern":[[22,2,1,"","forward"]],"slapo.pattern.Pattern":[[22,2,1,"","forward"]],"slapo.pipeline":[[23,4,1,"","analyze_tie_weights"]],"slapo.random":[[24,1,1,"","CudaRNGStatesTracker"],[24,4,1,"","get_cuda_rng_tracker"],[24,4,1,"","is_random_seed_set"],[24,4,1,"","model_parallel_cuda_manual_seed"],[24,4,1,"","set_random_seed"]],"slapo.random.CudaRNGStatesTracker":[[24,2,1,"","add"],[24,2,1,"","fork"],[24,2,1,"","get_states"],[24,2,1,"","reset"],[24,2,1,"","set_states"]],"slapo.schedule":[[26,1,1,"","Schedule"],[26,1,1,"","ScheduleMetadata"],[26,1,1,"","SubgraphWrapper"],[26,4,1,"","create_schedule"],[26,4,1,"","list_primitives"]],"slapo.schedule.Schedule":[[26,2,1,"","find"],[26,2,1,"","find_node"],[26,2,1,"","find_subgraph"],[26,2,1,"","trace_until"]],"slapo.schedule.SubgraphWrapper":[[26,2,1,"","forward"]],"slapo.tracer":[[27,4,1,"","trace"]],slapo:[[25,1,1,"","Database"],[25,3,1,"","FunctionType"],[25,1,1,"","ModulePattern"],[25,1,1,"","OrderedDict"],[25,1,1,"","Pattern"],[25,1,1,"","ScheduleMetadata"],[25,1,1,"","Space"],[25,1,1,"","Symbol"],[25,4,1,"","analyze_tie_weights"],[25,4,1,"","checkpoint"],[25,4,1,"","consolidate_model"],[25,4,1,"","create_schedule"],[25,4,1,"","dataclass"],[25,4,1,"","field"],[25,4,1,"","get_cuda_rng_tracker"],[25,4,1,"","get_dialect_cls"],[25,4,1,"","get_logger"],[25,4,1,"","init_empty_weights"],[25,4,1,"","init_target_engine"],[13,0,0,"-","initialization"],[25,4,1,"","is_random_seed_set"],[25,4,1,"","list_primitives"],[25,1,1,"","partial"],[22,0,0,"-","pattern"],[23,0,0,"-","pipeline"],[24,0,0,"-","random"],[25,4,1,"","register_primitive"],[26,0,0,"-","schedule"],[25,4,1,"","set_random_seed"],[25,4,1,"","trace"],[25,4,1,"","trace_module"],[27,0,0,"-","tracer"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","attribute","Python attribute"],"4":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:attribute","4":"py:function"},terms:{"0":[0,2,3,4,9,16,17,18,20,23,25],"00":[3,4],"001":3,"05":0,"08":4,"089":4,"09":4,"094":[2,4],"1":[0,2,3,9,16,17,23,25,26],"10":3,"100":3,"1024":[0,2,3],"102kb":3,"11":3,"12":3,"13":3,"14":3,"148":[0,4],"15":3,"16":[0,3],"17":3,"18":3,"181":[0,2],"19":3,"1e":[0,3],"2":[0,2,3,23,25,26],"20":[3,25],"2013":[24,25],"21":3,"22":3,"23":3,"3":[0,3],"30522":3,"3072":0,"4":3,"4096":3,"5":[3,20],"512":3,"526":25,"571":3,"6":3,"7":3,"8":[0,2,3],"847":[3,4],"9":3,"boolean":18,"break":2,"case":[0,25],"class":[0,2,9,11,16,17,18,20,21,22,24,25,26],"default":[0,2,3,17,24,25],"do":[0,2,16,20,24,25],"final":[0,2],"float":[16,20,21],"function":[0,2,3,7,9,11,13,14,16,17,18,20,21,22,23,24,25,26,27],"import":[0,2,3,29],"int":[9,16,20,21,23,24,25],"long":3,"new":[0,18,25,26],"return":[0,2,3,9,16,18,20,22,23,24,25,26],"static":[0,2,18,25,26],"super":[0,2],"true":[0,2,3,13,16,18,20,25,26],"try":[0,26],"while":[0,9,13,16,17,20,21,22,25,26],A:[1,2,4,6,9,13,16,20,21,25,26],As:[0,2],By:[0,2,3],For:[0,9,24,26,29],If:[0,2,16,18,20,23,25,26,29],In:[0,25],It:[0,6,18,25,26],NOT:[25,26],The:[0,2,3,9,16,18,20,21,23,25,26,29],Then:0,To:[0,2,20,29],_:[0,2,3],__annotations__:25,__hash__:25,__init__:[0,2,25],__repr__:25,_c:[0,2],_check:[0,2],_init_weight:3,_missing_typ:25,_nn:[0,2],_proj:0,_stacklevel:0,a_1:2,a_2:2,abil:26,about:[2,25],abov:[0,16],absolut:16,accept:[0,18,20],accordingli:2,achiev:0,across:[17,24,25],act_fn:20,activ:[2,18,20,21,25],actual:2,ad:[0,25],adamw:3,add:[0,2,24,25],add_1:0,add_2:0,addit:[0,2,20,25,26],address:6,affect:0,after:[0,2,3,9,24,25],afterward:[9,16,17,20,21,22,25,26],again:0,alia:[18,25],align:20,all:[0,9,11,13,16,17,18,20,21,22,23,24,25,26],all_reduc:2,allow_non:[11,25],along:2,alreadi:[0,2,3,23,25],also:[0,2,3,13,18,24,25,29],although:[9,16,17,20,21,22,25,26],alwai:[0,2,24,25],always_enable_tp_se:[24,25],among:24,an:[0,2,3,13,16,18,20,25],analyz:[9,23,25],analyze_tie_rank:9,analyze_tie_weight:[23,25],ani:[0,18,23,25,26,27],annot:[0,2],anoth:[0,2],apart:0,api:[0,3,6,15,26],appli:[0,3,14,16,25,26],applic:25,apply_and_build_schedul:3,apply_causal_mask:[0,16],apply_schedul:[3,14],approach:29,approxim:2,ar:[0,2,9,13,17,18,20,23,24,25],arbitrari:18,arg:[9,18,22,25],argument:[0,9,18,20,25,26],assign:[9,24,25],attach:[0,25],attent:[1,3,4,6,19],attention_mask:[0,3,16],attention_output:0,attn:0,attn_bia:16,attn_nam:16,attn_op_nam:[0,16],attn_pdrop:16,attn_prob:0,attn_sch:0,attn_scor:0,attribut:[0,2,18,25],auto:16,autoconfig:3,automat:18,avail:[13,25,26],awslab:29,axi:2,b:[2,3],b_1:2,b_2:2,back:16,backend:[0,2],backward:[2,3,18],base:[0,2,24,25],basic:[0,2,24],basin:2,batch_siz:[0,16,17],becaus:[25,26],becom:[0,2],been:[0,25],befor:[2,20,25],begin:[2,25],behavior:25,belong:26,below:[0,2],bert:3,bertattent:3,bertembed:3,bertencod:3,bertintermedi:3,bertlay:3,bertlmheadmodel:3,bertlmpredictionhead:3,bertmodel:3,bertonlymlmhead:3,bertoutput:3,bertpredictionheadtransform:3,bertselfattent:3,bertselfoutput:3,better:0,between:6,bf16:[3,16],bia:[0,2,3,16,18,20,21,26],bias_ge_lu_0:2,biasgelu:2,biasgelu_0:2,biasgelufunct:18,blow:[13,25],bmatrix:2,bodi:[0,2],bool:[13,16,20,21,23,24,25,26],both:[0,2,20,21,25],bs:[0,3],buffer:[13,25],build:3,bwd_post:2,c:[0,29],call:[0,2,3,9,16,17,20,21,22,24,25,26],call_modul:[0,26],callabl:[20,25,26],can:[0,2,3,16,18,24,26,29],candid:25,care:[2,9,16,17,20,21,22,25,26],cast:16,causal:16,cd:29,cdot:0,cfg_dict:25,cfg_dict_to_str:25,chang:[0,2,6,16,25,29],check:[0,2,16,24,25],checkpoint:[3,25],choos:0,ckpt_ratio:3,cl:[3,25],clear:25,clearli:0,clone:[3,25,29],cls_type:[11,25],code:[0,2,3,9],codebas:29,column:2,com:29,come:2,command:29,commit:[16,25],common:[0,6],commun:9,compar:[0,25],comparison:25,compat:[18,24],compil:[0,2],complex:2,compon:2,comput:[0,2,9,16,17,18,20,21,22,25,26],concrete_arg:0,conduct:[0,3],config:[3,9,25],configur:3,connect:[0,21],consequenti:0,consid:2,consist:[0,2],consolid:25,consolidate_model:25,constraint:[0,26],consum:0,contain:26,context:[13,18,25],contigu:0,control:0,convent:2,convert:[6,25],copi:[16,18,24,25],core:0,core_attn_subgraph:0,coreattent:[0,16],correct:[16,17],correctli:[0,2],correspond:[0,2,18,25],cover:[0,2],cpu:2,creat:[3,13,25,26],create_schedul:[0,2,25,26],create_symbol:25,creation:25,cross:[16,17],cross_entropi:19,ctx:18,cuda:[0,3,16,24,25],cudarngstatestrack:24,current:[0,18,20],custom:[3,20],cut:26,cutlass:[0,16],d:[16,25],d_k:0,data:[2,3,9,18,20,24,25],databas:25,dataclass:25,dataflow:[0,2,26],db:25,db_file_nam:25,debug:[0,2,25],debugg:0,decod:[3,9],decode_metadata:9,decompos:[0,2],decoupl:6,deep:0,deepspe:[10,25],deepspeed_pipe_engin:9,deepspeedpipestagewrapp:9,def:[0,2,3],default_factori:25,defin:[0,2,9,16,17,18,20,21,22,25,26],definit:6,defint:3,demonstr:0,dens:[0,3],dense_bia:0,dense_weight:0,depict:0,deriv:20,detail:[0,2,3,25],determin:[2,25],dev:29,develop:29,devic:[1,3,4,6,9,13,20,25,26],dialect:[11,25,26],dict:[9,23,25,26,27],dictionari:[24,25,26],differ:[0,2,23,24,25,26],differenti:18,dim:0,dimens:[0,2,17],direct:24,directli:[0,3,6,18,26,29],dispatch:[0,26],dist:[2,25,26],distribut:[2,17,25],documetn:16,doe:[0,21,25],doesn:[0,2],download:[0,2,3],dp_rank:[24,25],dropout:[0,3,16,18,20,21,24,25],dropout_mask:16,dropout_p:16,dtype:[0,3,20],due:25,dunder:25,dure:[2,18],e:[0,18,25,29],each:[0,2,9,18,20],easi:0,easier:26,easiest:29,easili:[0,2],effect:29,effici:[0,6,16,24],either:[0,2,18,25],element:[0,20,25],elementwise_affin:[0,3],els:25,embed:[3,16],emploi:0,empti:[0,2,13,25],enabl:[2,6,13,24,25],encod:[3,9],encode_metadata:9,encoder_attention_mask:16,encoder_hidden_st:16,encount:16,end:[0,2,3,25],enforc:18,engin:[8,9,10,25],enhanc:0,enough:2,entropi:17,env:2,ep:[0,3],eq:25,equival:[16,18],error:[16,25],estim:16,etc:16,even:[0,24,25],everi:[9,16,17,20,21,22,25,26],exactli:0,examin:25,exampl:[0,2,3,9,24],execut:[0,4,6],exist:25,exit:24,expect:2,explicitli:[0,2,3],express:[0,2,26],extra:20,extra_repr:20,f:[0,2,3,25],facebookresearch:16,factor:17,factori:[25,26],fals:[0,2,3,11,13,16,20,21,24,25],field:25,fifo:25,file:[4,25],find:[0,2,26],find_nod:26,find_subgraph:26,finish:0,first:[0,2,3,18],five:0,fix:25,fix_at:25,fixm:25,flag:[0,2],flash:[0,3,16],flash_attention_op_0:0,flash_attn:0,flash_attn_ref:16,flashattent:16,flashattentionop:[0,16],flashattentionop_0:0,flashattentiontriton:16,flat:9,flat_and_name_tensor_list:9,flatten:[0,2,9],flexibl:2,float16:3,floattensor:16,follow:[0,2,16,18,25,29],footprint:25,fork:[24,25],format:[9,26],former:[9,16,17,20,21,22,25,26],formula:18,forward:[0,2,9,16,17,18,20,21,22,25,26],found:[3,25],fp16:[3,9,16],fp32:16,frac:0,framework:[0,2,11,23,25,26],framework_dialect:12,from:[0,3,6,9,16,18,20,23,25,26,29],from_pretrain:3,fromkei:25,frozen:25,full:[0,2,3],func:25,functiontyp:[25,26],further:25,fuse:[0,2,3,18,20,21],fused_bia:19,fused_layer_norm_0:0,fused_linear:0,fused_qkv:[0,16],fused_qkv_0:0,fusedlayernorm:0,fusedlayernorm_0:0,fusedmlp:21,fusedqkv:[0,20],fusedqkv_0:0,fusion:0,futur:25,fuzzi:[0,2],fwd_post:2,fx:[0,9,26],g:[0,18],galleri:[0,2,3,4],gelu:[2,18,20,21],gelu_new:20,geluactiv:3,gemm:0,gener:[0,1,2,3,25,26],get:[0,11,16,24,25],get_all_dialect:11,get_cuda_rng_track:[24,25],get_dialect_cl:[11,25],get_logg:25,get_rank:2,get_simple_nested_list_str:9,get_stat:24,get_world_s:2,get_xfoemers_attn_op_by_nam:16,getattr_1:0,getattr_2:0,getattr_3:0,getattr_4:0,getitem:0,getitem_1:0,getitem_2:0,getitem_3:0,getitem_6:0,getitem_7:0,getitem_8:0,git:29,github:29,give:0,given:[9,18,25,26],gloo:2,go:[0,2,3],gpt:2,gpu:[0,2,4,9,24],grab:0,grad:18,grad_output:18,gradient:18,gradual:0,graph:[0,2,26],graph_modul:[0,2],graphmodul:[0,2,9,25,27],greatli:0,group:[2,17,20,24,25,26],guid:[0,2,3],ha:[0,16,18,25],half:2,handl:0,hash:[16,25],have:[0,2,3,18,21,23,24,25,26,29],head:[0,16,20],head_dim:16,head_mask:16,help:0,helper:9,here:[0,2,3,25],hf:16,hidden:[0,20,21],hidden_s:[0,2,16,17,20,21],hidden_st:[0,16,20,21],hierarch:[0,2],hierarchi:[0,25,26],high:16,hold:2,hook:[9,16,17,20,21,22,25,26],how:[0,2],howev:20,hs:0,http:29,hub:3,huggingfac:[3,16,18],id:[17,23,25],idea:0,ident:0,identifi:25,idx:25,ignor:[9,16,17,20,21,22,25,26],implement:[0,2,3,16,20,25],implicitli:0,in_featur:[0,2,3,20],includ:[0,2,9,25,26],include_buff:[13,25],incorpor:0,independ:0,index:[6,25],info:[0,2],inform:[9,20],init:25,init_ds_engin:7,init_empty_weight:[13,25],init_on_devic:13,init_target_engin:25,init_weight:[0,2,3],initi:[2,3,7,12,24,25],inject:3,inner:26,inp:18,inplac:[0,3,20],input:[0,2,3,16,18,20,21,24],input_id:3,input_tensor:0,insert:[2,25],insid:26,instal:[0,2,3,6],instanc:[0,2,9,16,17,20,21,22,25,26],instanti:[0,2],instead:[0,2,9,16,17,18,20,21,22,25,26],intend:18,inter:9,interfac:20,intermedi:[3,21],intermediate_act_fn:3,intermediate_s:21,invok:20,ipynb:[0,2,3],ir:0,is_fix:25,is_pipeline_partit:[23,25],is_random_seed_set:[24,25],item:[3,9,25],iter:25,its:[0,2,3,25],itself:6,jfc4050:16,jit:[0,2],json:3,jupyt:[0,2,3],just:[0,2,13,18,24,25],jvp:18,k:[0,16,25],k_proj:0,keep:0,kei:[0,3,25],kernel:[0,2,3,16,21],key_lay:16,key_padding_mask:16,keyerror:25,keyword:[9,25],kwarg:[7,9,18,25,26,27],label:[3,17],label_smooth:17,lack:25,lambda:26,languag:6,lanuch:2,larg:[3,6],last:25,later:[0,2,24],latter:[0,9,16,17,20,21,22,25,26],launch:0,layer:[2,3],layer_norm:0,layer_past:16,layernorm:[0,3],learn:0,left:[0,2],len:9,let:0,level:[0,2,23,25,26],leverag:[0,2,3],lib:[0,2],librari:0,lifo:25,like:[2,3,25],limit:16,line:[0,2,20],linear1:2,linear1_bia:2,linear1_weight:2,linear2:2,linear:[2,3,19,21],linearwithact:20,linearwithdropout:20,linearwithseparatebia:[0,2,20],linearwithsyncfunc:[2,20],list:[0,9,25,26],list_primit:[25,26],live:9,lm:[2,18,25],ln_pattern:0,ln_subgraph:0,load:[3,25],local:[0,2],log:[16,25],log_spac:25,logger:25,logit:17,loop:3,loss:[3,17],loss_fn:9,low:2,lower:16,lr:3,lve:3,machin:2,made:0,mai:[0,25,26],main:3,mainli:[9,16,25,26],maintain:9,make:[0,2,3,9,21,25],manag:[13,24,25],mani:18,manual_se:24,map:[23,25,26],mask:16,match:[0,2,23,25,26],math:16,mathrm:0,matmul:0,matmul_1:0,matrix:[0,2],max_sm:16,maximum:16,mb:4,mean:[2,24,25],megatron:[2,18,25],memori:[23,25],memory_efficient_fus:[20,21],merg:0,messag:16,meta:[13,25],metadata:[9,25,26],metadata_str:9,method:[9,16,17,18,20,21,22,24,25,26],micro_batch_s:17,min_sm:16,minimum:16,minut:[0,2,3],mlp:[1,4,6,19],mod:[0,2,9,26],mode:[2,18,25],model:[6,7,9,13,14,24,25,27],model_config:3,model_parallel_cuda_manual_se:24,model_schedul:[3,12],modifi:20,modul:[1,3,4,6,9,16,17,20,21,22,23,25,26,27],modulelist:3,modulepattern:[22,25],more:[0,2],most:[0,2,26],move:25,move_to_end:25,msg:16,much:0,multi:[1,4,6,20],multipl:[0,2],multipli:2,must:[17,18,25],n_head:0,name:[0,2,9,16,22,23,24,25,26],name_onli:[25,26],nativ:16,native_flash_attn:16,native_xform:[0,16],nccl:2,necessari:[0,2],need:[0,2,9,16,17,18,20,21,22,25,26],needs_input_grad:18,nest:9,new_gelu:18,new_shap:0,next:[0,25],nhead:16,nn:[0,2,3,9,20,23,25,26,27],node:[0,2,26],non:[0,2,18,24],none:[0,2,9,16,17,18,20,22,24,25,26],norm:0,note:[9,16,21,25,26],notebook:[0,2,3],noth:25,notic:[0,2],num_attention_head:16,num_head:20,number:[0,2,18,20,23,25],numer:16,nvfuser:[0,2],nvidia:0,object:[23,25,26],obtain:0,od:25,onc:16,one:[0,2,9,16,17,20,21,22,25,26],ones:3,onli:[0,2,16,24,25,26],op:[0,12],oper:[0,3,16,18,24],opt_model:[0,2,3],optim:[1,4,6,26],option:[16,24,25,26],order:[16,25],ordereddict:25,organ:16,orig_act:21,origin:[0,2,9,21,24,25,26],original_nam:[0,2],other:[0,2,18,25,26],otherwis:[25,26],our:[0,24],out:[0,2],out_featur:[0,2,3,20],outdens:16,outer:26,output:[0,2,3,9,16,17,18,20,29],output_attent:16,output_proj:16,over:0,overhead:0,overridden:[9,16,17,18,20,21,22,25,26],own:20,p:[0,3,16,20],packag:[0,2,3,29],padding_idx:3,pair:25,parallel:[3,17,20,24,25],parallelcrossentropi:17,param_init_fn:25,param_tag:[25,26],paramet:[0,3,9,13,16,17,20,21,23,24,25,26,27],parent:[25,26],part:[0,2,25],partial:[0,2,25],partit:[23,25],pass:[0,2,3,9,16,17,18,20,21,22,25,26],path:[9,25,26],pattern:[0,2,12,25,26],pattern_fn:26,pep:25,perceptron:2,perform:[0,2,3,9,16,17,18,20,21,22,24,25,26],permut:0,permute_1:0,permute_2:0,permute_for_scor:0,pip:29,pipelin:[8,10,12,24,25,26],pipelinemodul:9,place:20,pleas:[0,2,16,26],point:2,pointer:24,pop:25,popitem:25,posit:16,position_embed:3,pp_rank:[24,25],practic:0,predefin:[2,3],predict:3,prefix:3,present:25,preserv:[0,2,25,26],previou:2,previous:0,primit:[0,2,6,25,26],print:[0,2,3,9,20,25,29],print_read:[0,2],probabl:[0,16,20,21],process:[16,25,26],processgroup:[25,26],progress:[0,6],proj:0,proj_sch:0,project:16,properli:25,provid:[0,2,3,25,26,29],purpos:24,put:[13,25],py:[0,2,3,4],pypi:29,python3:[0,2],python:[0,2,3,6,29],pytorch:[0,2,6,16,25],q:[0,16],q_proj:0,qk:0,qkv:20,qkv_subgraph:0,queri:[0,3],query_lay:16,query_padding_mask:16,quick:[1,4,6],r:[0,18],rais:25,ram:[13,25],random:[12,25],rang:[3,17],rank:[2,9,17,24,25],ratio:3,re:[20,25],readabl:0,realiti:0,realiz:2,reason:0,recip:[9,16,17,20,21,22,25,26],recursivescriptmodul:[0,2],reduc:[0,25],reduce_forward_output:2,refer:0,reflect:0,regex:26,regex_or_pattern_fn:26,region:24,regist:[9,11,16,17,20,21,22,25,26],register_framework_dialect:11,register_primit:25,registr:11,registri:10,regular:[0,26],relat:9,reli:25,remain:0,rememb:25,remov:25,reorder:16,reorder_op:16,replac:[2,21,24,25,26],repr:25,repres:18,represent:20,reproduc:[24,25],requir:[0,18,20],reset:[24,25],reshape_for_scor:16,reshaped_qkv:0,resid_pdrop:[16,21],residu:[0,21,26],restor:[9,25],result:[2,25],ret:9,retriev:18,reus:0,revers:9,rich:25,right:[0,2],rng:[24,25],root:[25,26],row:2,run:[0,2,3,9,16,17,20,21,22,25,26,29],runtim:25,s:[2,3,25],same:[0,16,20,23,24,25,26],sampl:20,satisfi:[0,26],save:[18,25],save_for_backward:18,save_for_forward:18,scale:16,scaled_dot_product:0,sch:[0,2,3,25],sch_config:14,sch_metadata:9,schedul:[3,6,9,12,14,25],schedule_kei:14,schedulemetadata:[9,25,26],script:[0,2,3],seamlessli:0,second:[0,2,3],see:[0,2,25],seed:[24,25],self:[0,2,3,16],self_attn:0,self_output:0,selfattent:16,separ:[0,2],seq:0,seq_len:0,seq_length:3,seqlen_k:16,seqlen_q:16,sequenc:[24,25],sequence_length:17,set:[0,6,20,23,24,25,26],set_random_se:[24,25],set_stat:24,setdefault:25,setup:2,sever:2,shallow:25,shape:[0,16],shard:[0,2],share:[23,25],should:[0,9,16,17,18,20,21,22,23,24,25,26],show:[0,2],shown:[0,2],signatur:16,silent:[9,16,17,20,21,22,25,26],simpl:25,simpler:0,simpli:0,sinc:[0,2,9,16,17,20,21,22,25,26],singl:[1,2,4,6,20],site:[0,2],size:[20,21,24],slapo:[0,2,12,29],slice:0,sm:16,small:0,smooth:17,so:[0,3,21,23,24,25,26],softmax:[0,16],softmax_scal:16,sourc:[0,2,3,7,9,11,13,14,16,17,18,20,21,22,23,24,25,26,27,29],space:25,special:16,specif:[25,26],specifi:[0,2,13,25,26],sphinx:[0,1,2,3],split:[0,17],sqrt:[0,16],squeez:0,stage:[9,23,25,26],stage_id:9,stage_id_2_arg_nam:9,stage_modul:9,start:[1,4,24],state:[0,24,25],step:3,still:[0,2,3],store:[18,25,26],str:[9,16,20,21,23,25,26,27],strategi:2,string:[9,20,21,25],structur:[0,9,25,26],sub:26,subclass:[9,16,17,18,20,21,22,25,26],subgraph:[0,2,26],subgraphwrapp:26,submodul:[0,23,25,26],subschedul:0,subsequ:18,successfulli:29,suffix:9,sugar:26,support:[0,2,16,20,23,25],sure:[0,2,3,9],symbol:25,sync:[2,20],sync_fn:[2,20],sync_op_or_fn:2,synchron:2,syntax:26,system:[0,2],t1:9,t2:9,t:[0,2,18],take:[0,9,16,17,20,21,22,25,26,29],target:[11,17,25,26],tension:6,tensor:[0,3,9,16,17,18,20,24,25],test:16,thei:[2,18,24,25],them:[0,9,16,17,20,21,22,25,26],therefor:[13,25],thi:[0,2,3,9,13,16,17,18,20,21,22,23,24,25,26],thing:0,those:0,though:[0,18],three:0,through:[0,3,29],thu:[0,6],ti:[9,23,25],tie:[23,25],tie_group:[23,25],tie_weight:[25,26],tie_weight_group:9,time:[0,2,3],to_dict:25,todo:25,togeth:0,token_type_embed:3,token_type_id:3,top:[0,23,25,26],top_mod:[23,25],topolog:[9,25],torch:[0,2,3,9,13,16,17,20,23,24,25],torchscript:[0,2,20,21],total:[0,2,3,4],total_stag:9,tp:[24,25],tp_rank:[24,25],trace:[0,2,23,25,26,27],trace_modul:25,trace_until:26,tracer:[0,12],track:[24,25],tracker:[24,25],tradit:0,train:[0,3,6],training_script_arg:25,transfer:[25,26],transform:[0,2,3,18],transform_act_fn:3,transpos:[0,2,16],transpose_for_scor:16,triangular:16,triton:[0,16],truediv:0,tunabl:25,tune:25,tupl:[16,18,23,25,26],tutori:[0,2],two:[0,2,23,24,25,29],type:[0,2,3,9,16,18,20,22,23,24,25,26],uncas:3,under:[13,25],unflatten:9,union:[25,26],unsafe_hash:25,until:26,upcast:16,updat:25,update_space_fn:25,us:[0,2,3,6,9,13,16,18,20,21,24,25,26],usabl:6,use_cach:16,use_reentr:25,use_torchscript:[20,21],user:[0,2,3,6,21],userwarn:[0,2],usr:[0,2],usual:[0,2,3],v100:0,v:[0,16,25],v_proj:0,val:25,valid:16,validate_sm_vers:16,valu:[0,3,9,18,25],value_lay:16,verif:[25,26],verifi:29,version:16,vertic:0,view:[0,25],view_1:0,view_2:0,vjp:18,vocab:17,vocab_parallel_cross_entropi:17,vocab_parallel_logit:17,w:18,wa:25,wai:[0,29],walk:3,want:[0,2,26],warn:[0,2,16],warning_onc:16,we:[0,2,3,9,16,24,25,26,29],weight:[0,2,9,20,23,25],were:18,when:[13,16,17,24,25,26,29],where:[0,2],whether:[2,13,16,18,20,21,23,25],which:[0,2,13,25],whose:26,wise:0,within:[9,16,17,20,21,22,25,26],without:[6,9,16],word_embed:3,work:0,world_siz:[2,20],would:[13,25],wrap:[0,2,9],wrappedtypecod:9,wrapper:[16,21,23],write:[0,2],x:[0,2,16,20,26],xa:2,xa_1:2,xa_2:2,xformer:[0,16],xformers_ref:16,yet:25,you:[0,2,3,16,18,20,26,29],your:20,zero:20},titles:["Optimize Attention Module on A Single Device","Gallery","Optimize MLP Module on Multi-Device","Quick Start","Computation times","Index","Slapo Documentation","slapo.framework_dialect.deepspeed.engine","slapo.framework_dialect.deepspeed","slapo.framework_dialect.deepspeed.pipeline","slapo.framework_dialect","slapo.framework_dialect.registry","Python API","slapo.initialization","slapo.model_schedule.api","slapo.model_schedule","slapo.op.attention","slapo.op.cross_entropy","slapo.op.fused_bias","slapo.op","slapo.op.linear","slapo.op.mlp","slapo.pattern","slapo.pipeline","slapo.random","slapo","slapo.schedule","slapo.tracer","Gallery","Installation"],titleterms:{A:0,api:[12,14],attent:[0,16],build:[0,2],comput:4,creat:[0,2],cross_entropi:17,deepspe:[7,8,9],definit:[0,2],devic:[0,2],document:6,dot:0,engin:7,framework_dialect:[7,8,9,10,11],fused_bia:18,fusion:2,galleri:[1,28],get:6,index:5,initi:13,instal:29,layer:0,linear:[0,20],mlp:[2,21],model:[0,2,3],model_schedul:[14,15],modul:[0,2],multi:2,op:[16,17,18,19,20,21],oper:2,optim:[0,2,3],parallel:2,pattern:22,pipelin:[9,23],product:0,project:0,python:12,pytorch:3,qkv:0,quick:3,random:24,refer:6,registri:11,replac:0,scale:0,schedul:[0,2,26],selfattent:0,singl:0,slapo:[3,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27],start:[3,6],submodul:[8,10,15,19],tensor:2,time:4,tracer:27,tutori:6}})