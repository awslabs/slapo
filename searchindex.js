Search.setIndex({docnames:["gallery/attention-single-gpu","gallery/debug-print","gallery/index","gallery/mlp-multi-gpu","gallery/quick-start","gallery/sg_execution_times","genindex","index","python_api/framework_dialect/deepspeed/engine","python_api/framework_dialect/deepspeed/index","python_api/framework_dialect/deepspeed/pipeline","python_api/framework_dialect/index","python_api/framework_dialect/registry","python_api/index","python_api/initialization","python_api/model_schedule/api","python_api/model_schedule/index","python_api/op/attention","python_api/op/cross_entropy","python_api/op/fused_bias","python_api/op/index","python_api/op/linear","python_api/op/mlp","python_api/pattern","python_api/pipeline","python_api/random","python_api/root","python_api/schedule","python_api/tracer","scripts/README","setup/index"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":5,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.intersphinx":1,"sphinx.ext.todo":2,"sphinx.ext.viewcode":1,sphinx:56},filenames:["gallery/attention-single-gpu.rst","gallery/debug-print.rst","gallery/index.rst","gallery/mlp-multi-gpu.rst","gallery/quick-start.rst","gallery/sg_execution_times.rst","genindex.rst","index.rst","python_api/framework_dialect/deepspeed/engine.rst","python_api/framework_dialect/deepspeed/index.rst","python_api/framework_dialect/deepspeed/pipeline.rst","python_api/framework_dialect/index.rst","python_api/framework_dialect/registry.rst","python_api/index.rst","python_api/initialization.rst","python_api/model_schedule/api.rst","python_api/model_schedule/index.rst","python_api/op/attention.rst","python_api/op/cross_entropy.rst","python_api/op/fused_bias.rst","python_api/op/index.rst","python_api/op/linear.rst","python_api/op/mlp.rst","python_api/pattern.rst","python_api/pipeline.rst","python_api/random.rst","python_api/root.rst","python_api/schedule.rst","python_api/tracer.rst","scripts/README.rst","setup/index.rst"],objects:{"":[[26,0,0,"-","slapo"]],"slapo.Database":[[26,2,1,"","commit"],[26,2,1,"","load"]],"slapo.ModulePattern":[[26,2,1,"","forward"]],"slapo.OrderedDict":[[26,2,1,"","clear"],[26,2,1,"","copy"],[26,2,1,"","fromkeys"],[26,2,1,"","items"],[26,2,1,"","keys"],[26,2,1,"","move_to_end"],[26,2,1,"","pop"],[26,2,1,"","popitem"],[26,2,1,"","setdefault"],[26,2,1,"","update"],[26,2,1,"","values"]],"slapo.Pattern":[[26,2,1,"","forward"]],"slapo.Space":[[26,2,1,"","cfg_dict_to_str"],[26,2,1,"","clone"],[26,2,1,"","create_symbol"],[26,2,1,"","log_space"],[26,2,1,"","next"],[26,2,1,"","reset"],[26,2,1,"","to_dict"]],"slapo.Symbol":[[26,2,1,"","add"],[26,2,1,"","fix_at"],[26,2,1,"","is_fixed"]],"slapo.framework_dialect":[[12,0,0,"-","registry"]],"slapo.framework_dialect.deepspeed":[[8,0,0,"-","engine"],[10,0,0,"-","pipeline"]],"slapo.framework_dialect.deepspeed.engine":[[8,4,1,"","init_ds_engine"]],"slapo.framework_dialect.deepspeed.pipeline":[[10,1,1,"","DeepSpeedPipeStageWrapper"],[10,1,1,"","WrappedTypeCode"],[10,4,1,"","analyze_tie_ranks"],[10,4,1,"","decode_metadata"],[10,4,1,"","deepspeed_pipe_engine"],[10,4,1,"","encode_metadata"],[10,4,1,"","flat_and_name_tensor_list"],[10,4,1,"","flatten"],[10,4,1,"","get_simple_nested_list_str"],[10,4,1,"","unflatten"]],"slapo.framework_dialect.deepspeed.pipeline.DeepSpeedPipeStageWrapper":[[10,2,1,"","forward"]],"slapo.framework_dialect.registry":[[12,4,1,"","get_all_dialects"],[12,4,1,"","get_dialect_cls"],[12,4,1,"","register_framework_dialect"]],"slapo.initialization":[[14,4,1,"","init_empty_weights"],[14,4,1,"","init_on_device"]],"slapo.model_schedule":[[15,0,0,"-","api"]],"slapo.model_schedule.api":[[15,4,1,"","apply_schedule"]],"slapo.op":[[17,0,0,"-","attention"],[18,0,0,"-","cross_entropy"],[19,0,0,"-","fused_bias"],[21,0,0,"-","linear"],[22,0,0,"-","mlp"]],"slapo.op.attention":[[17,1,1,"","FlashAttention"],[17,1,1,"","FlashAttentionOp"],[17,4,1,"","flash_attn_ref"],[17,4,1,"","get_xfoemers_attn_op_by_name"],[17,4,1,"","validate_sm_version"],[17,4,1,"","warning_once"],[17,4,1,"","xformers_ref"]],"slapo.op.attention.FlashAttention":[[17,2,1,"","forward"],[17,2,1,"","reshape_for_scores"]],"slapo.op.attention.FlashAttentionOp":[[17,2,1,"","forward"]],"slapo.op.cross_entropy":[[18,1,1,"","ParallelCrossEntropy"],[18,4,1,"","vocab_parallel_cross_entropy"]],"slapo.op.cross_entropy.ParallelCrossEntropy":[[18,2,1,"","forward"]],"slapo.op.fused_bias":[[19,1,1,"","BiasGeLUFunction"],[19,4,1,"","new_gelu"]],"slapo.op.fused_bias.BiasGeLUFunction":[[19,2,1,"","backward"],[19,2,1,"","forward"]],"slapo.op.linear":[[21,1,1,"","FusedQKV"],[21,1,1,"","LinearWithAct"],[21,1,1,"","LinearWithDropout"],[21,1,1,"","LinearWithSeparateBias"],[21,1,1,"","LinearWithSyncFunc"]],"slapo.op.linear.FusedQKV":[[21,2,1,"","forward"]],"slapo.op.linear.LinearWithAct":[[21,2,1,"","extra_repr"],[21,2,1,"","forward"]],"slapo.op.linear.LinearWithDropout":[[21,2,1,"","extra_repr"],[21,2,1,"","forward"]],"slapo.op.linear.LinearWithSeparateBias":[[21,2,1,"","forward"]],"slapo.op.linear.LinearWithSyncFunc":[[21,2,1,"","extra_repr"],[21,2,1,"","forward"]],"slapo.op.mlp":[[22,1,1,"","FusedMLP"]],"slapo.op.mlp.FusedMLP":[[22,2,1,"","forward"]],"slapo.partial":[[26,3,1,"","args"],[26,3,1,"","func"],[26,3,1,"","keywords"]],"slapo.pattern":[[23,1,1,"","ModulePattern"],[23,1,1,"","Pattern"]],"slapo.pattern.ModulePattern":[[23,2,1,"","forward"]],"slapo.pattern.Pattern":[[23,2,1,"","forward"]],"slapo.pipeline":[[24,4,1,"","analyze_tie_weights"]],"slapo.random":[[25,1,1,"","CudaRNGStatesTracker"],[25,4,1,"","get_cuda_rng_tracker"],[25,4,1,"","is_random_seed_set"],[25,4,1,"","model_parallel_cuda_manual_seed"],[25,4,1,"","set_random_seed"]],"slapo.random.CudaRNGStatesTracker":[[25,2,1,"","add"],[25,2,1,"","fork"],[25,2,1,"","get_states"],[25,2,1,"","reset"],[25,2,1,"","set_states"]],"slapo.schedule":[[27,1,1,"","Schedule"],[27,1,1,"","ScheduleMetadata"],[27,1,1,"","SubgraphWrapper"],[27,4,1,"","create_schedule"],[27,4,1,"","list_primitives"]],"slapo.schedule.Schedule":[[27,2,1,"","find"],[27,2,1,"","find_node"],[27,2,1,"","find_subgraph"],[27,2,1,"","named_schedules"],[27,2,1,"","trace_until"]],"slapo.schedule.SubgraphWrapper":[[27,2,1,"","forward"]],"slapo.tracer":[[28,4,1,"","trace"]],slapo:[[26,1,1,"","Database"],[26,3,1,"","FunctionType"],[26,1,1,"","ModulePattern"],[26,1,1,"","OrderedDict"],[26,1,1,"","Pattern"],[26,1,1,"","ScheduleMetadata"],[26,1,1,"","Space"],[26,1,1,"","Symbol"],[26,1,1,"","Verify"],[26,4,1,"","analyze_tie_weights"],[26,4,1,"","checkpoint"],[26,4,1,"","consolidate_model"],[26,4,1,"","create_schedule"],[26,4,1,"","dataclass"],[26,4,1,"","field"],[26,4,1,"","get_cuda_rng_tracker"],[26,4,1,"","get_dialect_cls"],[26,4,1,"","get_logger"],[26,4,1,"","init_empty_weights"],[26,4,1,"","init_target_engine"],[14,0,0,"-","initialization"],[26,4,1,"","is_random_seed_set"],[26,4,1,"","list_primitives"],[26,1,1,"","partial"],[23,0,0,"-","pattern"],[24,0,0,"-","pipeline"],[25,0,0,"-","random"],[26,4,1,"","register_primitive"],[27,0,0,"-","schedule"],[26,4,1,"","set_random_seed"],[26,4,1,"","trace"],[26,4,1,"","trace_module"],[28,0,0,"-","tracer"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","attribute","Python attribute"],"4":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:attribute","4":"py:function"},terms:{"0":[0,1,3,4,5,10,17,18,19,21,24,26],"00":[4,5],"001":4,"0044":1,"0061":1,"016":[1,5],"0345":1,"0400":1,"0465":1,"05":0,"07":5,"0785":1,"0797":1,"081":[3,5],"1":[0,1,3,4,10,17,18,24,26,27],"10":4,"100":4,"1024":[0,3,4],"1049":1,"11":4,"1102":1,"12":4,"122":[4,5],"1241":1,"128":[0,5],"13":4,"14":4,"142kb":4,"1495":1,"15":4,"1566":1,"16":[0,4],"1668":1,"17":4,"18":4,"181":[0,3],"19":4,"1999":1,"1e":[0,4],"2":[0,1,3,4,24,26,27],"20":[4,26],"2013":[25,26],"2093":1,"21":4,"2112":1,"22":4,"2212":1,"23":4,"2419":1,"2423":1,"2517":1,"2561":1,"2624":1,"2633":1,"2643":1,"2nd":1,"3":[0,4],"30522":4,"3058":1,"3066":1,"3072":0,"3130":1,"3448":1,"3451":1,"348":5,"3rd":1,"4":[1,4],"4095":1,"4096":4,"4139":1,"4231":1,"4362":1,"4388":1,"4416":1,"4432":1,"4597":1,"4734":1,"4840":1,"5":[4,21],"5021":1,"512":[3,4],"526":26,"5374":1,"5644":1,"571":4,"5825":1,"5938":1,"6":4,"7":4,"8":[0,3,4],"8882":1,"9":4,"9146":1,"9303":1,"9335":1,"boolean":19,"break":3,"case":[0,26],"class":[0,1,3,10,12,17,18,19,21,22,23,25,26,27],"default":[0,3,4,18,25,26],"do":[0,3,17,21,25,26],"final":[0,3],"float":[17,21,22],"function":[0,1,3,4,8,10,12,14,15,17,18,19,21,22,23,24,25,26,27,28],"import":[0,1,3,4,30],"int":[10,17,21,22,24,25,26],"long":4,"new":[0,19,26,27],"return":[0,1,3,4,10,17,19,21,23,24,25,26,27],"static":[0,3,19,26,27],"super":[0,1,3],"true":[0,3,4,14,17,19,21,26,27],"try":[0,27],"while":[0,10,14,17,18,21,22,23,26,27],A:[2,3,5,7,10,14,17,21,22,26,27],And:1,As:[0,3],By:[0,3,4],For:[0,3,10,25,27,30],If:[0,3,17,19,21,24,26,27,30],In:[0,1,26],It:[0,1,7,19,26,27],NOT:[26,27],On:1,One:1,The:[0,1,3,4,10,17,19,21,22,24,26,27,30],Then:0,To:[0,1,3,21,30],_:[0,1,3,4],__annotations__:26,__hash__:26,__init__:[0,1,3,26],__repr__:26,_c:[0,3],_check:[0,3],_init_weight:4,_missing_typ:26,_nn:[0,3],_proj:0,_stacklevel:0,a_1:3,a_2:3,abil:27,about:[3,26],abov:[0,1,17],absolut:17,accept:[0,19,21],accordingli:3,achiev:0,across:[18,25,26],act_fn:21,activ:[1,3,19,21,22,26],actual:3,ad:[0,26],adamw:4,add:[0,1,3,25,26],add_1:0,add_2:0,addit:[0,3,21,26,27],address:7,affect:0,after:[0,1,3,4,10,25,26],afterward:[10,17,18,21,22,23,26,27],again:[0,1],alia:[19,26],align:21,all:[0,10,12,14,17,18,19,21,22,23,24,25,26,27],all_reduc:3,allow_non:[12,26],along:3,alreadi:[0,1,3,4,24,26],also:[0,1,3,4,14,19,25,26,30],although:[1,10,17,18,21,22,23,26,27],alwai:[0,1,3,25,26],always_enable_tp_se:[25,26],among:25,an:[0,1,3,4,14,17,19,21,26,27],analyz:[10,24,26],analyze_tie_rank:10,analyze_tie_weight:[24,26],ani:[0,19,24,26,27,28],annoi:1,annot:[0,3],anoth:[0,3],apart:0,api:[0,4,7,16,27],appli:[0,4,15,17,26,27],applic:26,apply_and_build_schedul:4,apply_causal_mask:[0,17],apply_schedul:[4,15],approach:30,approxim:3,ar:[0,1,3,10,14,18,19,21,24,25,26],arbitrari:19,arg:[10,19,23,26],argument:[0,1,10,19,21,26,27],assign:[1,10,25,26],attach:[0,26],attent:[2,4,5,7,20],attention_mask:[0,4,17],attention_output:0,attn:0,attn_bia:17,attn_nam:17,attn_op_nam:[0,17],attn_pdrop:17,attn_prob:0,attn_sch:0,attn_scor:0,attribut:[0,3,19,26],auto:17,autoconfig:4,automat:19,avail:[14,26,27],awslab:30,axi:3,b:[3,4],b_1:3,b_2:3,back:17,backend:[0,3],backward:[3,4,19],base:[0,3,25,26],basic:[0,3,25],basin:3,batch_siz:[0,17,18],becaus:[1,26,27],becom:[0,1,3],been:[0,26],befor:[3,21,26],begin:[3,26],behavior:26,belong:27,below:[0,3],bert:4,bertattent:4,bertembed:4,bertencod:4,bertintermedi:4,bertlay:4,bertlmheadmodel:4,bertlmpredictionhead:4,bertmodel:4,bertonlymlmhead:4,bertoutput:4,bertpredictionheadtransform:4,bertselfattent:4,bertselfoutput:4,better:0,between:7,bf16:[4,17],bia:[0,3,4,17,19,21,22,27],bias_ge_lu_0:3,biasgelu:3,biasgelu_0:3,biasgelufunct:19,blow:[14,26],bmatrix:3,bodi:[0,3],bool:[14,17,21,22,24,25,26,27],both:[0,3,21,22,26,27],bs:[0,4],buffer:[14,26],build:[1,4],bwd_post:3,c:[0,30],call:[0,3,4,10,17,18,21,22,23,25,26,27],call_modul:[0,27],callabl:[21,26,27],can:[0,1,3,4,17,19,25,27,30],candid:26,cannot:1,care:[3,10,17,18,21,22,23,26,27],cast:17,causal:17,cd:30,cdot:0,cfg_dict:26,cfg_dict_to_str:26,chang:[0,3,7,17,26,30],check:[0,3,17,25,26],checkpoint:[4,26],choos:0,ckpt_ratio:4,cl:[4,26],clear:26,clearli:0,clone:[4,26,30],cls_type:[12,26],code:[0,1,3,4,10],codebas:30,column:3,com:30,come:3,command:30,commit:[17,26],common:[0,7],commun:10,compar:[0,3,26],comparison:26,compat:[19,25],compil:[0,3],complex:3,compon:3,comput:[0,1,3,10,17,18,19,21,22,23,26,27],concrete_arg:0,conduct:[0,4],config:[4,10,26],configur:4,connect:[0,22],consequenti:0,consid:3,consist:[0,1,3],consolid:26,consolidate_model:26,constraint:[0,27],consum:[0,1],contain:27,context:[3,14,19,26],contigu:0,control:0,convent:3,convert:[7,26],copi:[17,19,25,26],core:0,core_attn_subgraph:0,coreattent:[0,17],correct:[1,3,17,18],correctli:[0,3],correspond:[0,3,19,26],cover:[0,3],cpu:3,creat:[1,4,14,26,27],create_schedul:[0,1,3,26,27],create_symbol:26,creation:26,cross:[17,18],cross_entropi:20,ctx:19,cuda:[0,4,17,25,26],cudarngstatestrack:25,current:[0,19,21,27],custom:[1,4,21],cut:27,cutlass:[0,17],d:[17,26],d_k:0,data:[1,3,4,10,19,21,25,26],databas:26,dataclass:26,dataflow:[0,1,3,27],db:26,db_file_nam:26,dead:1,debug:[0,2,3,5,7,26],debugg:0,decod:[4,10],decode_metadata:10,decompos:[0,3],decoupl:7,deep:0,deepspe:[11,26],deepspeed_pipe_engin:10,deepspeedpipestagewrapp:10,def:[0,1,3,4],default_factori:26,defin:[0,1,3,10,17,18,19,21,22,23,26,27],definit:7,defint:4,demonstr:0,dens:[0,4],dense_bia:0,dense_weight:0,depict:0,deriv:21,detail:[0,3,4,26],determin:[3,26],dev:30,develop:30,devic:[2,4,5,7,10,14,21,26,27],dialect:[12,26,27],dict:[10,24,26,27,28],dictionari:[25,26,27],differ:[0,3,24,25,26,27],differenti:19,dim:0,dimens:[0,3,18],direct:25,directli:[0,4,7,19,27,30],disappear:1,dispatch:[0,27],dist:[3,26,27],distribut:[3,18,26],documetn:17,doe:[0,22,26],doesn:[0,3],download:[0,1,3,4],dp_rank:[25,26],dropout:[0,4,17,19,21,22,25,26],dropout_mask:17,dropout_p:17,dtype:[0,4,21],due:26,dunder:26,dure:[1,3,19],e:[0,1,19,26,30],each:[0,3,10,19,21],easi:0,easier:27,easiest:30,easili:[0,3],effect:30,effici:[0,7,17,25],either:[0,3,19,26],element:[0,21,26],elementwise_affin:[0,4],els:26,embed:[4,17],emploi:0,empti:[0,3,14,26],enabl:[3,7,14,25,26],encapsul:3,encod:[4,10],encode_metadata:10,encoder_attention_mask:17,encoder_hidden_st:17,encount:17,end:[0,1,3,4,26],enforc:19,engin:[9,10,11,26],enhanc:0,enough:3,entropi:18,env:3,ep:[0,4],eq:26,equival:[17,19],error:[3,17,26],estim:17,etc:17,evalu:1,even:[0,1,25,26],everi:[10,17,18,21,22,23,26,27],exactli:0,examin:26,exampl:[0,1,3,4,10,25],example_input:[3,26],execut:[0,1,3,5,7],exist:26,exit:25,expect:3,explicitli:[0,3,4],express:[0,3,27],extra:21,extra_repr:21,f:[0,1,3,4,26],facebookresearch:17,factor:18,factori:[26,27],fals:[0,1,3,4,12,14,17,21,22,25,26],feel:1,field:26,fifo:26,file:[5,26],find:[0,3,27],find_nod:27,find_subgraph:27,finish:0,first:[0,1,3,4,19],five:0,fix:[1,26],fix_at:26,fixm:26,flag:[0,3],flash:[0,4,17],flash_attention_op_0:0,flash_attn:0,flash_attn_ref:17,flashattent:17,flashattentionop:[0,17],flashattentionop_0:0,flashattentiontriton:17,flat:10,flat_and_name_tensor_list:10,flatten:[0,3,10],flexibl:3,float16:4,floattensor:17,follow:[0,3,17,19,26,30],footprint:26,fork:[25,26],format:[10,27],former:[10,17,18,21,22,23,26,27],formula:19,forward:[0,1,3,10,17,18,19,21,22,23,26,27],found:[4,26],fp16:[4,10,17],fp32:17,frac:0,framework:[0,3,12,24,26,27],framework_dialect:13,from:[0,1,4,7,10,17,19,21,24,26,27,30],from_pretrain:4,fromkei:26,frozen:26,full:[0,1,3,4],func:26,functiontyp:[26,27],further:[3,26],fuse:[0,3,4,19,21,22],fused_bia:20,fused_layer_norm_0:0,fused_linear:0,fused_qkv:[0,17],fused_qkv_0:0,fusedlayernorm:0,fusedlayernorm_0:0,fusedmlp:22,fusedqkv:[0,21],fusedqkv_0:0,fusion:0,futur:26,fuzzi:[0,3],fwd_post:3,fx:[0,1,10,27],g:[0,19],galleri:[0,1,3,4,5],gelu:[1,3,19,21,22],gelu_new:21,geluactiv:4,gemm:0,gener:[0,1,2,3,4,26,27],get:[0,12,17,25,26],get_all_dialect:12,get_cuda_rng_track:[25,26],get_dialect_cl:[12,26],get_logg:26,get_rank:3,get_simple_nested_list_str:10,get_stat:25,get_world_s:3,get_xfoemers_attn_op_by_nam:17,getattr_1:[0,1],getattr_2:0,getattr_3:0,getattr_4:0,getitem:0,getitem_1:0,getitem_2:0,getitem_3:0,getitem_6:0,getitem_7:0,getitem_8:0,git:30,github:30,give:0,given:[10,19,26,27],gloo:3,go:[0,1,3,4],gpt:3,gpu:[0,3,5,10,25],grab:0,grad:19,grad_fn:1,grad_output:19,gradient:19,gradual:0,graph:[0,1,3,27],graph_modul:[0,3],graphmodul:[0,1,3,10,26,28],greatli:0,group:[3,18,21,25,26,27],guarante:3,guid:[0,3,4],ha:[0,1,17,19,26],half:3,hand:1,handl:0,hash:[17,26],have:[0,1,3,4,19,22,24,25,26,27,30],head:[0,17,21],head_dim:17,head_mask:17,help:0,helper:10,here:[0,1,3,4,26],hf:17,hidden:[0,21,22],hidden_s:[0,1,3,17,18,21,22],hidden_st:[0,17,21,22],hierarch:[0,3],hierarchi:[0,26,27],high:17,hold:3,hook:[10,17,18,21,22,23,26,27],how:[0,1,3],howev:[1,21],hs:0,http:30,hub:4,huggingfac:[4,17,19],i:1,id:[18,24,26],idea:0,ident:[0,1],identifi:26,idx:26,ignor:[10,17,18,21,22,23,26,27],implement:[0,3,4,17,21,26],implicitli:0,in_featur:[0,3,4,21],includ:[0,3,10,26,27],include_buff:[14,26],incorpor:0,incorrect:1,independ:0,index:[7,26],info:[0,3],inform:[10,21],init:26,init_ds_engin:8,init_empty_weight:[14,26],init_on_devic:14,init_target_engin:26,init_weight:[0,1,3,4],initi:[3,4,8,13,25,26],inject:4,inner:27,inp:19,inplac:[0,4,21],input:[0,3,4,17,19,21,22,25],input_id:4,input_tensor:0,insert:[3,26],insid:27,instal:[0,1,3,4,7],instanc:[0,3,10,17,18,21,22,23,26,27],instanti:[0,3],instead:[0,1,3,10,17,18,19,21,22,23,26,27],intend:19,inter:10,interfac:21,intermedi:[1,4,22],intermediate_act_fn:4,intermediate_s:22,invok:21,ipynb:[0,1,3,4],ir:[0,1],is_fix:26,is_pipeline_partit:[24,26],is_random_seed_set:[25,26],item:[4,10,26],iter:[26,27],its:[0,1,3,4,26],itself:[7,27],jfc4050:17,jit:[0,3],json:4,jupyt:[0,1,3,4],just:[0,3,14,19,25,26],jvp:19,k:[0,17,26],k_proj:0,keep:0,kei:[0,4,26],kernel:[0,3,4,17,22],key_lay:17,key_padding_mask:17,keyerror:26,keyword:[10,26],kwarg:[8,10,19,26,27,28],label:[4,18],label_smooth:18,lack:26,lambda:27,languag:7,lanuch:3,larg:[4,7],last:26,later:[0,3,25],latter:[0,10,17,18,21,22,23,26,27],launch:0,layer:[1,3,4],layer_norm:0,layer_past:17,layernorm:[0,4],lazili:1,leaf:1,learn:0,left:[0,3],len:10,let:[0,1],level:[0,3,24,26,27],leverag:[0,3,4],lib:[0,3],librari:0,lifo:26,like:[1,3,4,26],limit:17,line:[0,3,21],linear1:[1,3],linear1_bia:3,linear1_weight:3,linear2:[1,3],linear:[1,3,4,20,22],linearwithact:21,linearwithdropout:21,linearwithseparatebia:[0,3,21],linearwithsyncfunc:[3,21],list:[0,10,26,27],list_primit:[26,27],live:10,lm:[3,19,26],ln_pattern:0,ln_subgraph:0,load:[4,26],local:[0,3],log:[17,26],log_spac:26,logger:26,logit:18,look:1,loop:4,loss:[4,18],loss_fn:10,low:3,lower:17,lr:4,lve:4,machin:3,made:0,mai:[0,1,26,27],main:4,mainli:[10,17,26,27],maintain:10,make:[0,1,3,4,10,22,26],manag:[14,25,26],mani:19,manual_se:25,map:[24,26,27],mark:1,mask:17,match:[0,3,24,26,27],math:17,mathrm:0,matmul:0,matmul_1:0,matrix:[0,3],max_sm:17,maximum:17,mb:5,mean:[1,3,25,26],megatron:[3,19,26],memori:[24,26],memory_efficient_fus:[21,22],mention:1,merg:0,messag:17,meta:[14,26],metadata:[10,26,27],metadata_str:10,method:[10,17,18,19,21,22,23,25,26,27],micro_batch_s:18,min_sm:17,minimum:17,minut:[0,1,3,4],mlp:[1,2,5,7,20],mlpwithprint:1,mlpwithwrongprint:1,mod:[0,1,3,10,27],mode:[3,19,26],model:[1,7,8,10,14,15,25,26,28],model_config:4,model_parallel_cuda_manual_se:25,model_schedul:[4,13],modifi:21,modul:[1,2,4,5,7,10,17,18,21,22,23,24,26,27,28],modulelist:4,modulepattern:[23,26],more:[0,3],most:[0,3,27],move:26,move_to_end:26,msg:17,much:0,multi:[2,5,7,21],multipl:[0,3],multipli:3,must:[18,19,26],n:1,n_head:0,name:[0,3,10,17,23,24,25,26,27],name_onli:[26,27],named_schedul:27,nativ:17,native_flash_attn:17,native_xform:[0,17],nccl:3,necessari:[0,3],need:[0,3,10,17,18,19,21,22,23,26,27],needs_input_grad:19,nest:10,new_gelu:19,new_shap:0,next:[0,1,26],nhead:17,nn:[0,1,3,4,10,21,24,26,27,28],node:[0,3,27],non:[0,3,19,25],none:[0,1,3,10,17,18,19,21,23,25,26,27],norm:0,normal:1,note:[10,17,22,26,27],notebook:[0,1,3,4],noth:26,notic:[0,1,3],now:1,num_attention_head:17,num_head:21,number:[0,3,19,21,24,26],numer:[1,3,17],nvalu:1,nvfuser:[0,3],nvidia:0,object:[24,26,27],obtain:0,od:26,onc:17,one:[0,3,10,17,18,21,22,23,26,27],ones:4,onli:[0,1,3,17,25,26,27],op:[0,1,13],oper:[0,1,4,17,19,25],opt_model:[0,3,4],optim:[2,5,7,27],option:[17,25,26,27],order:[17,26],ordereddict:26,organ:17,orig_act:22,origin:[0,3,10,22,25,26,27],original_nam:[0,3],other:[0,1,3,19,26,27],otherwis:[1,26,27],our:[0,1,25],out:[0,1,3],out_featur:[0,3,4,21],outdens:17,outer:27,output:[0,1,3,4,10,17,18,19,21,30],output_attent:17,output_proj:17,over:[0,27],overhead:0,overridden:[10,17,18,19,21,22,23,26,27],own:21,p:[0,4,17,21],packag:[0,1,3,4,30],padding_idx:4,pair:26,parallel:[4,18,21,25,26],parallelcrossentropi:18,param_init_fn:26,param_tag:[26,27],paramet:[0,4,10,14,17,18,21,22,24,25,26,27,28],parent:[26,27],part:[0,1,3,26],partial:[0,3,26],partit:[24,26],pass:[0,3,4,10,17,18,19,21,22,23,26,27],path:[10,26,27],pattern:[0,3,13,26,27],pattern_fn:27,pep:26,perceptron:3,perform:[0,3,4,10,17,18,19,21,22,23,25,26,27],permut:0,permute_1:0,permute_2:0,permute_for_scor:0,pip:30,pipelin:[9,11,13,25,26,27],pipelinemodul:10,place:21,pleas:[0,3,17,27],point:[1,3],pointer:25,pop:26,popitem:26,posit:17,position_embed:4,pp_rank:[25,26],practic:0,predefin:[3,4],predict:4,prefix:[4,27],present:26,preserv:[0,1,3,26,27],previou:3,previous:0,primit:[0,3,7,26,27],prine:1,print1:1,print2:1,print3:1,print:[0,2,3,4,5,7,10,21,26,30],print_1:1,print_2:1,print_read:[0,3],probabl:[0,17,21,22],problem:1,process:[17,26,27],processgroup:[26,27],progress:[0,7],proj:0,proj_sch:0,project:17,properli:[1,26],provid:[0,1,3,4,26,27,30],proxi:1,purpos:25,put:[14,26],py:[0,1,3,4,5],pypi:30,python3:[0,3],python:[0,1,3,4,7,30],pytorch:[0,1,3,7,17,26],q:[0,17],q_proj:0,qk:0,qkv:21,qkv_subgraph:0,queri:[0,4],query_lay:17,query_padding_mask:17,quick:[2,5,7],r:[0,19],rais:26,ram:[14,26],randn:[1,3],random:[13,26],rang:[4,18],rank:[3,10,18,25,26],ratio:4,re:[21,26],readabl:0,realiti:0,realiz:3,reason:[0,1],recip:[10,17,18,21,22,23,26,27],recursivescriptmodul:[0,3],reduc:[0,26],reduce_forward_output:3,refer:0,reflect:0,regex:27,regex_or_pattern_fn:27,region:25,regist:[10,12,17,18,21,22,23,26,27],register_framework_dialect:12,register_primit:26,registr:12,registri:11,regular:[0,27],relat:10,reli:26,remain:0,rememb:26,remov:[1,26],reorder:17,reorder_op:17,replac:[3,22,25,26,27],report:3,repr:26,repres:19,represent:21,reproduc:[25,26],requir:[0,19,21],reset:[25,26],reshape_for_scor:17,reshaped_qkv:0,resid_pdrop:[17,22],residu:[0,22,27],restor:[10,26],result:[3,26],ret:10,retriev:19,reus:0,revers:10,rich:26,right:[0,3],rng:[25,26],root:[26,27],row:3,run:[0,1,3,4,10,17,18,21,22,23,26,27,30],runtim:[1,26],s:[1,3,4,26],same:[0,17,21,24,25,26,27],sampl:21,satisfi:[0,27],save:[19,26],save_for_backward:19,save_for_forward:19,scale:17,scaled_dot_product:0,sch:[0,1,3,4,26],sch_config:15,sch_metadata:10,schedul:[1,4,7,10,13,15,26],schedule_kei:15,schedulemetadata:[10,26,27],script:[0,1,3,4],seamlessli:0,second:[0,1,3,4],see:[0,1,3,26],seed:[25,26],self:[0,1,3,4,17],self_attn:0,self_output:0,selfattent:17,separ:[0,3],seq:0,seq_len:0,seq_length:4,seqlen_k:17,seqlen_q:17,sequenc:[25,26],sequence_length:18,set:[0,7,21,24,25,26,27],set_random_se:[25,26],set_stat:25,setdefault:26,setup:3,sever:3,shallow:26,shape:[0,1,17],shard:[0,3],share:[24,26],should:[0,10,17,18,19,21,22,23,24,25,26,27],show:[0,1,3],shown:[0,3],signatur:17,silent:[10,17,18,21,22,23,26,27],simpl:26,simpler:0,simpli:0,sinc:[0,3,10,17,18,21,22,23,26,27],singl:[2,3,5,7,21],site:[0,3],size:[1,21,22,25],slapo:[0,1,3,13,30],slice:0,sm:17,small:0,smooth:18,so:[0,1,4,22,24,25,26,27],softmax:[0,17],softmax_scal:17,solv:1,some:1,sourc:[0,1,3,4,8,10,12,14,15,17,18,19,21,22,23,24,25,26,27,28,30],space:26,special:17,specif:[1,26,27],specifi:[0,1,3,14,26,27],sphinx:[0,1,2,3,4],split:[0,18],sqrt:[0,17],squeez:0,stage:[10,24,26,27],stage_id:10,stage_id_2_arg_nam:10,stage_modul:10,start:[1,2,5,25],state:[0,25,26],statement:1,step:4,still:[0,1,3,4],store:[19,26,27],str:[1,10,17,21,22,24,26,27,28],strategi:3,string:[1,10,21,22,26],structur:[0,10,26,27],sub:[1,27],subclass:[10,17,18,19,21,22,23,26,27],subgraph:[0,3,27],subgraphwrapp:27,submodul:[0,24,26,27],subschedul:[0,27],subsequ:19,successfulli:30,suffix:10,sugar:27,support:[0,3,17,21,24,26],sure:[0,1,3,4,10],symbol:26,sync:[3,21],sync_fn:[3,21],sync_op_or_fn:3,synchron:3,syntax:27,system:[0,3],t1:10,t2:10,t:[0,1,3,19],take:[0,10,17,18,21,22,23,26,27,30],target:[12,18,26,27],tension:7,tensor:[0,1,4,10,17,18,19,21,25,26],test:17,thei:[3,19,25,26],them:[0,10,17,18,21,22,23,26,27],therefor:[1,14,26],thi:[0,1,3,4,10,14,17,18,19,21,22,23,24,25,26,27],thing:0,those:[0,3],though:[0,19],three:0,through:[0,4,30],thu:[0,1,7],ti:[10,24,26],tie:[24,26],tie_group:[24,26],tie_weight:[26,27],tie_weight_group:10,time:[0,1,3,4],to_dict:26,todo:26,togeth:0,token_type_embed:4,token_type_id:4,top:[0,24,26,27],top_mod:[24,26],topolog:[10,26],torch:[0,1,3,4,10,14,17,18,21,24,25,26],torchscript:[0,3,21,22],total:[0,1,3,4,5],total_stag:10,tp:[25,26],tp_rank:[25,26],trace:[0,1,3,24,26,27,28],trace_modul:26,trace_until:27,tracer:[0,1,13],track:[25,26],tracker:[25,26],tradit:0,train:[0,4,7],training_script_arg:26,transfer:[26,27],transform:[0,3,4,19],transform_act_fn:4,transpos:[0,3,17],transpose_for_scor:17,treat:1,triangular:17,triton:[0,17],truediv:0,tunabl:26,tune:26,tupl:[17,19,24,26,27],turori:1,tutori:[0,1,3],two:[0,1,3,24,25,26,30],type:[0,3,4,10,17,19,21,23,24,25,26,27],uncas:4,under:[14,26],unflatten:10,union:[26,27],unsafe_hash:26,until:27,upcast:17,updat:26,update_space_fn:26,us:[0,1,3,4,7,10,14,17,19,21,22,25,26,27],usabl:7,usag:1,use_cach:17,use_reentr:26,use_torchscript:[21,22],user:[0,3,4,7,22],userwarn:[0,3],usr:[0,3],usual:[0,3,4],v100:0,v:[0,17,26],v_proj:0,val:26,valid:17,validate_sm_vers:17,valu:[0,1,4,10,19,26],value_lay:17,verif:[26,27],verifi:[3,26,30],version:17,vertic:0,view:[0,1,26],view_1:0,view_2:0,viewbackward0:1,vjp:19,vocab:18,vocab_parallel_cross_entropi:18,vocab_parallel_logit:18,w:19,wa:26,wai:[0,30],walk:4,want:[0,1,3,27],warn:[0,3,17],warning_onc:17,we:[0,1,3,4,10,17,25,26,27,30],weight:[0,3,10,21,24,26],weird:1,well:27,were:19,when:[1,14,17,18,25,26,27,30],where:[0,3],whether:[3,14,17,19,21,22,24,26],which:[0,1,3,14,26],whose:27,wise:0,within:[10,17,18,21,22,23,26,27],without:[7,10,17],won:1,word_embed:4,work:[0,1],world_siz:[3,21],would:[14,26],wrap:[0,3,10],wrappedtypecod:10,wrapper:[17,22,24],write:[0,3],x:[0,3,17,21,27],xa:3,xa_1:3,xa_2:3,xformer:[0,17],xformers_ref:17,yet:26,yield:27,you:[0,1,3,4,17,19,21,27,30],your:21,zero:21},titles:["Optimize Attention Module on A Single Device","Debugging with Print","Gallery","Optimize MLP Module on Multi-Device","Quick Start","Computation times","Index","Slapo Documentation","slapo.framework_dialect.deepspeed.engine","slapo.framework_dialect.deepspeed","slapo.framework_dialect.deepspeed.pipeline","slapo.framework_dialect","slapo.framework_dialect.registry","Python API","slapo.initialization","slapo.model_schedule.api","slapo.model_schedule","slapo.op.attention","slapo.op.cross_entropy","slapo.op.fused_bias","slapo.op","slapo.op.linear","slapo.op.mlp","slapo.pattern","slapo.pipeline","slapo.random","slapo","slapo.schedule","slapo.tracer","Gallery","Installation"],titleterms:{A:0,api:[13,15],attent:[0,17],build:[0,3],comput:5,creat:[0,3],cross_entropi:18,debug:1,deepspe:[8,9,10],definit:[0,3],devic:[0,3],document:7,dot:0,engin:8,framework_dialect:[8,9,10,11,12],fused_bia:19,fusion:3,galleri:[2,29],get:7,index:6,initi:14,instal:30,layer:0,linear:[0,21],mlp:[3,22],model:[0,3,4],model_schedul:[15,16],modul:[0,3],multi:3,op:[17,18,19,20,21,22],oper:3,optim:[0,3,4],parallel:3,pattern:23,pipelin:[10,24],print:1,product:0,project:0,python:13,pytorch:4,qkv:0,quick:4,random:25,refer:7,registri:12,replac:0,scale:0,schedul:[0,3,27],selfattent:0,singl:0,slapo:[4,7,8,9,10,11,12,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28],start:[4,7],submodul:[9,11,16,20],tensor:3,time:5,tracer:28,tutori:7}})