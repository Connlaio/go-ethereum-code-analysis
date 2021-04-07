eth的源码又下面几个包

- downloader 		主要用于和网络同步，包含了传统同步方式和快速同步方式
- fetcher			主要用于基于块通知的同步，接收到当我们接收到NewBlockHashesMsg消息得时候，我们只收到了很多Block的hash值。 需要通过hash值来同步区块。
- filter			提供基于RPC的过滤功能，包括实时数据的同步(PendingTx)，和历史的日志查询(Log filter)
- gasprice			提供gas的价格建议， 根据过去几个区块的gasprice，来得到当前的gasprice的建议价格


eth 协议部分源码分析

- [以太坊的网络协议大概流程](eth以太坊协议分析.md)

fetcher部分的源码分析

- [fetch部分源码分析](eth-fetcher源码分析.md)

downloader 部分源码分析
	
- [节点快速同步算法](以太坊fast%20sync算法.md)
- [用来提供下载任务的调度和结果组装 queue.go](eth-downloader-queue.go源码分析.md)
- [用来代表对端，提供QoS等功能 peer.go](eth-downloader-peer源码分析.md)
- [快速同步算法 用来提供Pivot point的 state-root的同步 statesync.go](eth-downloader-statesync.md)
- [同步的大致流程的分析 ](eth-downloader源码分析.md)

filter 部分源码分析

- [提供布隆过滤器的查询和RPC过滤功能](eth-bloombits和filter源码分析.md)

> go-etherreum-master
>   |- accounts     /* 实现了高层级Ethereum账号管理 */
>   |    |- abi        // 该包实现了Ethereum的ABI(应用程序二进制接口) 
>   |         bind.go  // 该包生成Ethereum合约的Go绑定 
>   |    |- keystore   // 实现了Secp256k1私钥的加密存储 
>   |    |- usbwallet  // 该包实现了支持USB硬件钱包
>   |    accounts.go   // 定义了账号的结构
>   |- build
>   |- cmd         /* 命令行工具 */
>   |    |- abigen    // 一个源代码生成器，它将Ethereum智能合约定义(代码) 转换为易于使用的、编译时类型安全的Go package。
>   |    |- bootnode  // 该节点为Ethereum发现协议运行一个引导节点。
>   |    |- clef      // Clef可以用来签署交易和数据，并且可以代替geth的账户管理。这使DApps不依赖于geth的账户管理。
>   |    |- ethkey    // 该包描述了与以太坊keyfiles的交互命令行
>   |    |- evm       // 执行EVM代码片段的命令行
>   |    |- faucet    // 以太坊支持的轻量级客户
>   |    |- geth      // 启动客户端命令行
>   |    |- internal  // 提供与用户浏览器交互的工具
>   |    |- p2psim    // 客户端命令行模拟 HTTP API
>   |    |- puppeth   // 组装和维护个人网路的命令行
>   |    |- rlpdump   // 打印出兼顾用户友好和机器友好的RLP格式数据 
>   |    |- swarm     // bzzhash命令，用来更好的计算出swarm哈希树
>   |    |- utils     // 为go-ethereum命令提供说明
>   |    |- wnode     // 一个简单的Whisper节点
>   |- common      /* 一些工具函数 */
>   |    |- bitutil   // 快速位操作
>   |    |- compiler  // 该包包装了可执行的solidity编译器
>   |    |- hexutil   // 以0x为前缀的十六进制编码
>   |    |- math      // 数学工具
>   |    big.go       // 大整数
>   |    bytes.go     // bytes-hex工具
>   |    format.go    // 格式化time.Duration值
>   |    types.go     // 数据类型及转换
>   |- consensus   /* 以太坊的共识引擎 */
>   |    |- clique    // 实现了POA共识引擎
>   |    |- ethash    // 实现了POW共识引擎
>   |    |- misc      // 与DAO硬分叉相关的确认与共识
>   |    consensus.go // 定义了ChainReader和Engine接口 
>   |- console     /* 是一个Javascript解释的运行环境 */
>   |- contracts   /* 实现了支票簿合约和ENS  */
>   |    |- chequebook // “支票簿”可以创建并签署从单一合同到多个受益人的支票。它是对等微支付的传出支付处理程序。
>   |    |- ens        // ENS(ethereum name service)
>   |- core        /* 以太坊的核心数据结构和算法(虚拟机，状态，区块链，布隆过滤器) */
>   |    |- asm        // 汇编和反汇编接口
>   |    |- bloombits  // 过滤数据 
>   |    |- rawdb      // 
>   |    |- state      // 世界状态的实现
>   |    |    database.go
>   |    |    iterator.go
>   |    |    journal.go
>   |    |    managed_state.go
>   |    |    state_object.go
>   |    |    statedb.go
>   |    |    sync.go
>   |    |- types      // 区块链中的数据类型
>   |    |    block.go
>   |    |    bloom9.go
>   |    |    derive_sha.go
>   |    |    gen_header_json.go
>   |    |    gen_log_json.go
>   |    |    gen_receipt_json.go
>   |    |    gen_tx_json.go
>   |    |    log.go
>   |    |    receipt.go
>   |    |    transaction.go
>   |    |    transaction_signing.go
>   |    |- vm        // 实现evm
>   |    |
>   |    block_validator.go    
>   |    blockchain.go
>   |    block.go
>   |    chain_indexer.go
>   |    chain_makers.go
>   |    error.go
>   |    events.go
>   |    evm.go
>   |    gaspool.go
>   |    gen_genesis.go
>   |    genesis.go
>   |    genesis_alloc.go
>   |    headerchain.go
>   |    state_processor.go
>   |    state_transition.go
>   |    tx_cacher.go
>   |    tx_journal.go
>   |    tx_list.go
>   |    tx_pool.go
>   |    types.go
>   |- crypto      /* 加密和hash算法 */
>   |- dashboard   /* 仪表板是集成到geth的数据可视化工具，用于收集和可视化Ethereum节点的有用信息。 */
>   |- eth         /* 实现所有以太坊协议 */ 
>   |    |- downloader    // 手动全链同步
>   |    |- fetcher       // 基于块通知的同步
>   |    |- filter        // 用于区块、交易和日志事件的过滤
>   |    |- gasprice      // 返回区块的建议gasprice
>   |    |- tracers       // 收集JavaScript交易追踪
>   |    backend.go
>   |    bloombits.go
>   |    handler.go
>   |    metrics.go
>   |    peer.go
>   |    protocol.go
>   |    sync.go
>   |- ethclient   /* 提供以太坊的RPC客户端 */
>   |- ethdb       /* eth的数据库(包括实际使用的leveldb和供测试使用的内存数据库) */
>   |- ethstats    /* 提供网络状态报告 */
>   |- event       /* 处理实时事件 */
>   |- internal    /*  */ 
>   |- les         /* 轻量级Ethereum子协议 */
>   |- light       /* 实现按需检索能力的状态和链对象 */
>   |- log         /* 提供对人机均友好的日志信息 */
>   |- metrics     /* go-Metrics，为系统某服务做监控、统计 */
>   |- miner       /* 提供以太坊的区块创建和挖矿 */
>   |- node        /* 以太坊的多种类型的节点 */
>   |- p2p         /* 实现p2p网络协议 */
>   |- params      /* 以太坊系统中所用到的一些常量和变量 */
>   |- rlp         /* 以太坊序列化处理 */
>   |- rpc         /* 远程方法调用 */
>   |- signer
>   |- swarm       /* swarm 是一个分布式存储平台和内容分发服务 */
>   |- trie        /* 定义了以太坊重要的数据结构：Package trie implements Merkle Patricia Tries*/
>   |- wisper      /* 实现一种点对点的隐秘消息传输网络 */
>   |
>   interface.go   /* 定义了以太坊必要的接口，包括区块链读取、交易读取、链状态读取、同步、消息调用、过滤日志、设置gasPrice等 */
> ————————————————
> 版权声明：本文为CSDN博主「佛系布偶」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
> 原文链接：https://blog.csdn.net/lj900911/article/details/83449858