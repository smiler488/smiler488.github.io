// A title may be a plain string when it reads the same in both locales
// (GitHub, arXiv, Nature), or an { en, zh } pair when it needs translating.
const link = (title, url, category) => ({
  title: typeof title === "string" ? { en: title, zh: title } : title,
  url,
  category,
});

const t = (en, zh) => ({ en, zh });

export const navigatorGroups = [
  {
    id: "work",
    label: { en: "Work", zh: "工作" },
    hint: {
      en: "Campus systems, postdoc applications, funding and policy",
      zh: "校内系统、博士后申请、项目与政策",
    },
  },
  {
    id: "study",
    label: { en: "Research", zh: "学习科研" },
    hint: {
      en: "Literature, journals, data platforms and daily research tools",
      zh: "文献、期刊、数据平台与日常科研工具",
    },
  },
  {
    id: "daily",
    label: { en: "Daily", zh: "日常" },
    hint: {
      en: "Site building, developer consoles and everyday downtime",
      zh: "网站建设、开发后台与日常休闲",
    },
  },
];

export const navigatorCategories = [
  {
    id: "shzu",
    group: "work",
    label: { en: "Shihezi University", zh: "石河子大学" },
    description: {
      en: "Campus portals for graduate study, research administration, teaching, facilities, and finance.",
      zh: "研究生、科研管理、教学、设备与财务等校内常用入口。",
    },
    keywords: ["shzu", "campus", "graduate", "石河子", "研究生", "校内"],
  },
  {
    id: "cau-postdoc",
    group: "work",
    label: { en: "CAU & postdoc systems", zh: "中国农大与博士后" },
    description: {
      en: "China Agricultural University entries plus the postdoc application, fellowship, and recruitment systems.",
      zh: "中国农业大学相关入口，以及博士后进站、基金与招聘系统。",
    },
    keywords: ["CAU", "postdoc", "中国农业大学", "博士后", "进站", "招聘"],
  },
  {
    id: "funding-policy",
    group: "work",
    label: { en: "Funding, policy & IP", zh: "项目、政策与知识产权" },
    description: {
      en: "Grant systems, ministries, academies, standards, and patent offices used for applications and reporting.",
      zh: "基金系统、部委机构、学会院所、标准与专利入口，用于申报与材料准备。",
    },
    keywords: ["NSFC", "grant", "policy", "patent", "基金", "政策", "专利"],
  },
  {
    id: "literature",
    group: "study",
    label: { en: "Literature & discovery", zh: "文献与检索" },
    description: {
      en: "Search, request, organize, and follow papers—plus academic seminars and conference calendars.",
      zh: "文献检索、求助、管理与追踪，以及学术会议和讲座入口。",
    },
    keywords: ["papers", "search", "ablesci", "文献", "检索", "科研通", "会议"],
  },
  {
    id: "journals",
    group: "study",
    label: { en: "Journals & submission", zh: "期刊与投稿" },
    description: {
      en: "Target journals across plant science, phenomics, agriculture, and remote sensing, with submission helpers.",
      zh: "植物科学、表型组学、农业与遥感方向的目标期刊，以及投稿辅助工具。",
    },
    keywords: ["journal", "submission", "期刊", "投稿", "分区", "影响因子"],
  },
  {
    id: "data-platforms",
    group: "study",
    label: { en: "Data & compute platforms", zh: "科研数据与算力平台" },
    description: {
      en: "Agricultural, climate, soil, and satellite data gateways, model hubs, and hosted compute.",
      zh: "农业、气象、土壤与卫星数据入口，模型社区与在线算力平台。",
    },
    keywords: ["data", "climate", "satellite", "数据", "气象", "遥感", "算力"],
  },
  {
    id: "ai-models",
    group: "study",
    label: { en: "LLM & AI platforms", zh: "大模型与 AI 平台" },
    description: {
      en: "Official entry points for the large-model providers and scientific AI platforms used in day-to-day research.",
      zh: "日常科研中常用的大模型服务与科学智能平台官方入口，覆盖国内外主要厂商。",
    },
    keywords: ["LLM", "AI", "model", "大模型", "智能体", "人工智能", "API"],
  },
  {
    id: "toolbox",
    group: "study",
    label: { en: "Writing & tool box", zh: "写作与工具箱" },
    description: {
      en: "Manuscript writing, terminology, reference management, plotting, and small utilities used every week.",
      zh: "论文写作、术语查询、文献管理、绘图与日常小工具。",
    },
    keywords: ["writing", "plots", "zotero", "写作", "绘图", "工具"],
  },
  {
    id: "build-ship",
    group: "daily",
    label: { en: "Site & development", zh: "站点与开发" },
    description: {
      en: "The consoles and documentation behind this site, WeChat mini programs, and side projects.",
      zh: "本站、微信小程序与个人项目所需的后台与开发文档。",
    },
    keywords: ["docusaurus", "github", "wechat", "站点", "开发", "小程序"],
  },
  {
    id: "life",
    group: "daily",
    label: { en: "Life & downtime", zh: "生活与娱乐" },
    description: {
      en: "Film, video, reading, and community stops for the end of a long working day.",
      zh: "影视、视频、阅读与社区，用于一天工作之后的放松。",
    },
    keywords: ["movies", "video", "community", "影视", "视频", "娱乐"],
  },
];

export const navigatorLinks = [
  // 工作 · 石河子大学
  link(
    t("Shihezi University", "石河子大学"),
    "https://www.shzu.edu.cn/",
    "shzu"
  ),
  link(
    t("Unified Service Center", "统一服务中心"),
    "https://one.shzu.edu.cn/EIP/nonlogin/user/index.htm",
    "shzu"
  ),
  link(
    t("Online Service Hall", "网上服务大厅"),
    "https://serv.shzu.edu.cn/home",
    "shzu"
  ),
  link(t("Graduate School", "研究生院"), "http://yjsh.shzu.edu.cn/", "shzu"),
  link(
    t("Graduate Management System", "研究生管理系统"),
    "http://gs.shzu.edu.cn/gmis5/home/login",
    "shzu"
  ),
  link(
    t("Degree Regulations", "学位管理规定"),
    "http://yjsh.shzu.edu.cn/glgd_9076/list.htm",
    "shzu"
  ),
  link(
    t("College of Agriculture", "农学院"),
    "http://nxy.shzu.edu.cn/",
    "shzu"
  ),
  link(
    t("Key Lab of Oasis Eco-agriculture", "绿洲生态农业重点实验室"),
    "http://nxy.shzu.edu.cn/sys/main.htm",
    "shzu"
  ),
  link(
    t("Office of Science & Technology", "科学技术处"),
    "http://kyc.shzu.edu.cn/",
    "shzu"
  ),
  link(
    t("Lab & Equipment Office", "实验设备处"),
    "http://sysbc.shzu.edu.cn/",
    "shzu"
  ),
  link(t("Finance Office", "财务处"), "https://jcc.shzu.edu.cn/", "shzu"),
  link(
    t("Academic Affairs Office", "教务处"),
    "http://jwc.shzu.edu.cn/",
    "shzu"
  ),
  link("Blackboard", "https://bb.shzu.edu.cn/", "shzu"),
  link(
    t("Campus Payment Platform", "校园统一支付平台"),
    "http://mxfjf.shzu.edu.cn/wsyh/main.aspx",
    "shzu"
  ),
  link(
    t("Campus Card Service", "校园卡服务"),
    "http://card.shzu.edu.cn/",
    "shzu"
  ),
  link(t("Network Center", "网络中心"), "http://nc.shzu.edu.cn/", "shzu"),
  link(
    t("Career Service Platform", "智慧就业服务平台"),
    "https://scc.shzu.edu.cn/",
    "shzu"
  ),
  link(
    t("Shixiaozhi AI Assistant", "石小智 AI 助手"),
    "https://icss.shzu.edu.cn/p/index.html#/",
    "shzu"
  ),

  // 工作 · 中国农大与博士后
  link(
    t("China Agricultural University", "中国农业大学"),
    "https://www.cau.edu.cn/",
    "cau-postdoc"
  ),
  link(
    t("CAU College of Agronomy", "中国农业大学农学院"),
    "http://cab.cau.edu.cn/",
    "cau-postdoc"
  ),
  link(
    t("CAU College of Biological Sciences", "中国农业大学生物学院"),
    "https://cbs.cau.edu.cn/",
    "cau-postdoc"
  ),
  link(
    t("CAU Talent Office", "中国农大人才工作办公室"),
    "https://rcb.cau.edu.cn/",
    "cau-postdoc"
  ),
  link(
    t("Postdoc Entry Checklist", "博士后进站材料清单"),
    "https://rcb.cau.edu.cn/art/2019/1/28/art_35799_645822.html",
    "cau-postdoc"
  ),
  link(
    t("China Postdoctoral Science Foundation", "中国博士后科学基金会"),
    "https://www.chinapostdoctor.org.cn/home",
    "cau-postdoc"
  ),
  link(
    t("China Postdoc Online System", "中国博士后网上办公系统"),
    "https://www.chinapostdoctor.org.cn/auth/login.html",
    "cau-postdoc"
  ),
  link(
    t("China Scholarship Council", "国家留学网"),
    "https://www.csc.edu.cn/",
    "cau-postdoc"
  ),
  link(
    t("CSC Study Abroad Platform", "国家公派留学管理信息平台"),
    "https://sa.csc.edu.cn/student/",
    "cau-postdoc"
  ),
  link(
    t("CSC Postdoc Program Guide", "国家公派博士后项目指南"),
    "https://www.csc.edu.cn/article/4051",
    "cau-postdoc"
  ),
  link(
    t("ScienceHR · Postdoc Jobs", "科学人才网 · 博士后招聘"),
    "https://www.sciencehr.net/html/bsh/hw/",
    "cau-postdoc"
  ),
  link(
    t("GaoxiaoJob · Academic Jobs", "高校人才网"),
    "https://www.gaoxiaojob.com/",
    "cau-postdoc"
  ),
  link("EURAXESS", "https://euraxess.ec.europa.eu/", "cau-postdoc"),
  link(
    t("Science & Technology Backyard", "全国科技小院服务管理平台"),
    "https://stb.mae.edu.cn/student/StudentAchievement/index.html",
    "cau-postdoc"
  ),

  // 工作 · 项目、政策与知识产权
  link(
    t("NSFC", "国家自然科学基金委员会"),
    "http://www.nsfc.gov.cn/",
    "funding-policy"
  ),
  link(
    t("NSFC ISIS System", "科学基金网络信息系统"),
    "https://grants.nsfc.gov.cn/pmpweb/login",
    "funding-policy"
  ),
  link(
    t("NSFC Knowledge Portal", "基金大数据知识管理服务门户"),
    "https://kd.nsfc.cn/",
    "funding-policy"
  ),
  link(
    t("Ministry of Science and Technology", "科学技术部"),
    "http://www.most.gov.cn/index.html",
    "funding-policy"
  ),
  link(
    t("Ministry of Agriculture and Rural Affairs", "农业农村部"),
    "http://www.moa.gov.cn/",
    "funding-policy"
  ),
  link(
    t("Ministry of Education", "教育部"),
    "http://www.moe.gov.cn/",
    "funding-policy"
  ),
  link(
    t("Xinjiang Education Department", "新疆维吾尔自治区教育厅"),
    "http://jyt.xinjiang.gov.cn/edu/index.shtml",
    "funding-policy"
  ),
  link(
    t("China Association for Science and Technology", "中国科学技术协会"),
    "https://www.cast.org.cn/",
    "funding-policy"
  ),
  link(
    t("Chinese Academy of Sciences", "中国科学院"),
    "https://www.cas.cn/",
    "funding-policy"
  ),
  link(
    t("Chinese Academy of Engineering", "中国工程院"),
    "https://www.cae.cn/",
    "funding-policy"
  ),
  link(
    t("Chinese Academy of Agricultural Sciences", "中国农业科学院"),
    "https://www.caas.cn/",
    "funding-policy"
  ),
  link(
    t("Global Research Project Database", "全球科研项目数据库"),
    "http://project.llas.ac.cn/",
    "funding-policy"
  ),
  link(
    t("CNIPA", "国家知识产权局"),
    "https://www.cnipa.gov.cn/",
    "funding-policy"
  ),
  link(
    t("CNIPA Patent Publication", "中国专利公布公告"),
    "http://epub.cnipa.gov.cn/Index",
    "funding-policy"
  ),
  link("Google Patents", "https://patents.google.com/", "funding-policy"),
  link(
    t("National Standards Full-text", "国家标准全文公开"),
    "http://openstd.samr.gov.cn/bzgk/gb/index",
    "funding-policy"
  ),
  link(
    t("Industry Standards Platform", "行业标准信息服务平台"),
    "https://hbba.sacinfo.org.cn/",
    "funding-policy"
  ),

  // 学习 · 文献与检索
  link(
    t("AbleSci", "科研通 AbleSci"),
    "https://www.ablesci.com/",
    "literature"
  ),
  link(
    t("AbleSci · Research Nav", "科研通 · 科研导航"),
    "https://www.ablesci.com/daohang",
    "literature"
  ),
  link(t("CNKI", "中国知网"), "https://www.cnki.net/", "literature"),
  link("X-MOL", "https://www.x-mol.com/", "literature"),
  link("Web of Science", "https://www.webofscience.com/", "literature"),
  link("Semantic Scholar", "https://www.semanticscholar.org/", "literature"),
  link("arXiv", "https://arxiv.org/", "literature"),
  link("bioRxiv", "https://www.biorxiv.org/", "literature"),
  link("alphaXiv", "https://www.alphaxiv.org/", "literature"),
  link("Connected Papers", "https://www.connectedpapers.com/", "literature"),
  link("SciSpace", "https://typeset.io/", "literature"),
  link("TXYZ", "https://www.txyz.ai/", "literature"),
  link("ResearchGate", "https://www.researchgate.net/", "literature"),
  link("ORCID", "https://orcid.org/", "literature"),
  link(
    t("National Science Library, CAS", "中国科学院文献情报中心"),
    "https://www.las.ac.cn/",
    "literature"
  ),
  link(
    t("CAS Journal Ranking", "中科院期刊分区表"),
    "https://www.fenqubiao.com/Landing.html",
    "literature"
  ),
  link(
    t("XR Scholar Journal Ranking", "新锐期刊分区表"),
    "https://www.xr-scholar.com/",
    "literature"
  ),
  link(
    t("SciPing Academic Nav", "科塔学术导航"),
    "https://site.sciping.com/cas.html",
    "literature"
  ),
  link(
    t("China Academic Conferences Online", "中国学术会议在线"),
    "http://www.meeting.edu.cn/",
    "literature"
  ),
  link(
    t("ScienceNet · Conferences", "科学网 · 会议"),
    "https://meeting.sciencenet.cn/",
    "literature"
  ),
  link(t("Koushare", "蔻享学术"), "https://www.koushare.com/", "literature"),
  link("EasyChair", "https://www.easychair.org/", "literature"),
  link(t("ScienceNet", "科学网"), "http://www.sciencenet.cn/", "literature"),
  link(
    t("Muchong Forum", "小木虫论坛"),
    "http://muchong.com/bbs/",
    "literature"
  ),

  // 学习 · 期刊与投稿
  link("Nature", "https://www.nature.com/", "journals"),
  link("Science", "https://www.science.org/", "journals"),
  link("PNAS", "https://www.pnas.org/", "journals"),
  link(
    "Annual Review of Plant Biology",
    "https://www.annualreviews.org/content/journals/arplant",
    "journals"
  ),
  link(
    "New Phytologist",
    "https://nph.onlinelibrary.wiley.com/journal/14698137",
    "journals"
  ),
  link("Plant Physiology", "https://academic.oup.com/plphys", "journals"),
  link(
    "Journal of Experimental Botany",
    "https://academic.oup.com/jxb",
    "journals"
  ),
  link(
    "Plant Phenomics",
    "https://spj.science.org/journal/plantphenomics",
    "journals"
  ),
  link("Plant Methods", "https://plantmethods.biomedcentral.com/", "journals"),
  link(
    "in silico Plants",
    "https://academic.oup.com/insilicoplants/",
    "journals"
  ),
  link(
    "Trends in Plant Science",
    "https://www.cell.com/trends/plant-science/home",
    "journals"
  ),
  link(
    "Remote Sensing of Environment",
    "https://www.sciencedirect.com/journal/remote-sensing-of-environment",
    "journals"
  ),
  link(
    "Journal of Integrative Agriculture",
    "https://www.chinaagrisci.com/Jwk_zgnykxen/EN/2095-3119/home.shtml",
    "journals"
  ),
  link(
    "Frontiers in Plant Science",
    "https://www.frontiersin.org/journals/plant-science",
    "journals"
  ),
  link("Bio-protocol", "https://bio-protocol.org/cn", "journals"),
  link(
    t("Smart Agriculture", "智慧农业（中英文）"),
    "http://www.smartag.net.cn/CN/2096-8094/home.shtml",
    "journals"
  ),
  link(
    t("Scientia Agricultura Sinica", "中国农业科学"),
    "https://www.chinaagrisci.com/CN/0578-1752/home.shtml",
    "journals"
  ),
  link(
    t("Cotton Science", "棉花学报"),
    "http://journal.cricaas.com.cn/Jweb_mhxb/CN/1002-7807/home.shtml",
    "journals"
  ),
  link(
    t("Chinese Journal of Plant Ecology", "植物生态学报"),
    "https://www.plant-ecology.com/CN/1005-264X/home.shtml",
    "journals"
  ),
  link(
    t("Plant Physiology Journal", "植物生理学报"),
    "http://www.plant-physiology.com/",
    "journals"
  ),
  link(
    "Elsevier JournalFinder",
    "https://journalfinder.elsevier.com/",
    "journals"
  ),
  link(
    t("DatAuthor", "DatAuthor 投稿指南"),
    "https://datauthor.com/",
    "journals"
  ),

  // 学习 · 数据与算力
  link(
    t("National Agriculture Science Data Center", "国家农业科学数据中心"),
    "https://www.agridata.cn/#/home",
    "data-platforms"
  ),
  link(
    t("Agricultural S&T Knowledge Platform", "农业科技知识服务平台"),
    "https://agri.nais.net.cn/index.html",
    "data-platforms"
  ),
  link(
    t("China Agricultural Big Data", "中国农业大数据"),
    "http://www.agdata.cn/",
    "data-platforms"
  ),
  link(
    t("China Meteorological Data Service Center", "国家气象科学数据中心"),
    "https://data.cma.cn/",
    "data-platforms"
  ),
  link(
    t("China Meteorological Administration", "中国气象局"),
    "http://www.cma.gov.cn/",
    "data-platforms"
  ),
  link("FAOSTAT", "https://www.fao.org/faostat/en/#data/FBS", "data-platforms"),
  link(
    "Harmonized World Soil Database",
    "https://www.fao.org/soils-portal/soil-survey/soil-maps-and-databases/harmonized-world-soil-database-v12/en/",
    "data-platforms"
  ),
  link(
    t("ISSAS Soil Data Center", "南京土壤所数据中心"),
    "https://soildata.issas.ac.cn/",
    "data-platforms"
  ),
  link("NOAA", "https://www.noaa.gov/", "data-platforms"),
  link(
    "NASA POWER",
    "https://power.larc.nasa.gov/data-access-viewer/",
    "data-platforms"
  ),
  link("OpenTopography", "https://opentopography.org/", "data-platforms"),
  link(
    "Google Earth Engine Code Editor",
    "https://code.earthengine.google.com/",
    "data-platforms"
  ),
  link("LP DAAC", "https://lpdaac.usgs.gov/product_search/", "data-platforms"),
  link(
    t("Resource & Environment Data Platform", "资源环境科学数据平台"),
    "https://www.resdc.cn/",
    "data-platforms"
  ),
  link("Kaggle", "https://www.kaggle.com/", "data-platforms"),
  link("Codabench", "https://www.codabench.org/", "data-platforms"),
  link("Hugging Face", "https://huggingface.co/", "data-platforms"),
  link("ModelScope", "https://www.modelscope.cn/home", "data-platforms"),
  link("Google Colab", "https://colab.research.google.com/", "data-platforms"),
  link("Zenodo", "https://zenodo.org/", "data-platforms"),

  // 学习 · 大模型与 AI 平台
  link(t("Zhipu AI", "智谱 AI"), "https://www.zhipuai.cn/zh", "ai-models"),
  link(t("Moonshot AI", "月之暗面"), "https://www.moonshot.cn", "ai-models"),
  link("DeepSeek", "https://www.deepseek.com", "ai-models"),
  link("MiniMax", "https://www.minimax.io", "ai-models"),
  link(t("Xiaomi MiMo", "小米 MiMo"), "https://mimo.mi.com", "ai-models"),
  link(
    t("ScienceOne (CAS)", "磐石 ScienceOne"),
    "https://www.scienceone.cn",
    "ai-models"
  ),
  link("OpenAI", "https://openai.com", "ai-models"),
  link("Anthropic", "https://www.anthropic.com", "ai-models"),
  link("Google AI", "https://ai.google", "ai-models"),

  // 学习 · 写作与工具箱
  link(
    t("Overleaf Templates", "Overleaf 模板库"),
    "https://www.overleaf.com/latex/templates/?nocdn=true",
    "toolbox"
  ),
  link(
    "Academic Phrasebank",
    "https://www.phrasebank.manchester.ac.uk/",
    "toolbox"
  ),
  link("Purdue OWL", "https://owl.purdue.edu/owl/index.html", "toolbox"),
  link("QuillBot", "https://www.quillbot.com/", "toolbox"),
  link(
    t("DeepL Translator", "DeepL 翻译"),
    "https://www.deepl.com/translator",
    "toolbox"
  ),
  link(
    t("CNKI Translation", "CNKI 翻译助手"),
    "https://dict.cnki.net/index",
    "toolbox"
  ),
  link(
    t("TermOnline", "术语在线"),
    "https://www.termonline.cn/index",
    "toolbox"
  ),
  link("Planteome", "https://planteome.org/", "toolbox"),
  link("CAS Source Index", "https://cassi.cas.org/search.jsp", "toolbox"),
  link(
    t("Zotero Chinese Community", "Zotero 中文社区"),
    "https://zotero-chinese.com/",
    "toolbox"
  ),
  link(
    t("Zotero Plugins", "Zotero 插件商店"),
    "https://zotero-chinese.github.io/zotero-plugins/#/",
    "toolbox"
  ),
  link("Mathpix", "https://mathpix.com/", "toolbox"),
  link("diagrams.net", "https://app.diagrams.net/", "toolbox"),
  link("Mermaid Live Editor", "https://mermaid.live/edit", "toolbox"),
  link("ProcessOn", "https://www.processon.com/diagrams", "toolbox"),
  link(
    t("Apache ECharts Examples", "Apache ECharts 示例"),
    "https://echarts.apache.org/examples/zh/index.html",
    "toolbox"
  ),
  link("The R Graph Gallery", "https://r-graph-gallery.com/", "toolbox"),
  link("From Data to Viz", "https://www.data-to-viz.com/", "toolbox"),
  link(
    t("Seaborn Gallery", "Seaborn 示例库"),
    "https://seaborn.pydata.org/examples/index.html",
    "toolbox"
  ),
  link(
    t("Tuzhidian Chart Guide", "图之典"),
    "http://www.tuzhidian.com/",
    "toolbox"
  ),
  link(
    t("Data Viz Catalogue", "数据可视化工具目录"),
    "https://datavizcatalogue.com/ZH/index.html",
    "toolbox"
  ),
  link("ChiPlot", "https://www.chiplot.online/", "toolbox"),
  link(t("Figdraw", "Figdraw 绘科研"), "https://www.figdraw.com/#/", "toolbox"),
  link(t("Coolors", "Coolors 配色"), "https://coolors.co/", "toolbox"),
  link(t("remove.bg", "remove.bg 抠图"), "https://www.remove.bg/zh", "toolbox"),
  link(
    t("Unit Conversion", "在线单位换算"),
    "http://www.unitconversion.org/",
    "toolbox"
  ),
  link(
    t("TUNA Mirror (Tsinghua)", "清华大学开源镜像站"),
    "https://mirrors.tuna.tsinghua.edu.cn/",
    "toolbox"
  ),

  // 日常 · 站点与开发
  link("GitHub", "https://github.com/", "build-ship"),
  link("GitHub Pages", "https://pages.github.com/", "build-ship"),
  link("Docusaurus", "https://docusaurus.io/", "build-ship"),
  link(
    t("Docusaurus Docs (zh)", "Docusaurus 中文文档"),
    "https://docusaurus.io/zh-CN/docs/",
    "build-ship"
  ),
  link(
    "Algolia DocSearch",
    "https://docsearch.algolia.com/docs/what-is-docsearch",
    "build-ship"
  ),
  link("Google Analytics", "https://analytics.google.com/", "build-ship"),
  link(
    t("Google Search Console", "Google 搜索控制台"),
    "https://search.google.com/search-console",
    "build-ship"
  ),
  link("Supabase", "https://supabase.com/dashboard", "build-ship"),
  link("Vercel", "https://vercel.com/dashboard", "build-ship"),
  link(
    t("Alibaba Cloud Console", "阿里云控制台"),
    "https://console.aliyun.com/",
    "build-ship"
  ),
  link(
    t("Oracle Cloud Console", "Oracle 云控制台"),
    "https://cloud.oracle.com/",
    "build-ship"
  ),
  link(
    t("WeChat Official Accounts", "微信公众平台"),
    "https://mp.weixin.qq.com/",
    "build-ship"
  ),
  link(
    t("WeChat Developer Platform", "微信开发者平台"),
    "https://developers.weixin.qq.com/platform",
    "build-ship"
  ),
  link(
    t("Mini Program Dev Docs", "微信小程序开发文档"),
    "https://developers.weixin.qq.com/miniprogram/dev/framework/",
    "build-ship"
  ),
  link(
    t("Mini Program Design Guide", "微信小程序设计指南"),
    "https://developers.weixin.qq.com/miniprogram/design/",
    "build-ship"
  ),
  link(
    t("HarmonyOS Quick Start", "HarmonyOS 快速入门"),
    "https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/start-with-ets-stage",
    "build-ship"
  ),
  link(
    t("ICP Filing System", "ICP 备案管理系统"),
    "https://beian.miit.gov.cn/",
    "build-ship"
  ),
  link(
    t("Tencent Cloud Console", "腾讯云控制台"),
    "https://console.cloud.tencent.com/",
    "build-ship"
  ),
  link("Stack Overflow", "https://stackoverflow.com/", "build-ship"),
  link(
    t("Markdown Guide", "Markdown 官方教程"),
    "https://markdown.com.cn/basic-syntax/",
    "build-ship"
  ),
  link("iconfont", "https://www.iconfont.cn/", "build-ship"),

  // 日常 · 生活与娱乐
  link(t("IkanBot", "爱看机器人"), "https://www1.ikanbot.com/", "life"),
  link(t("Douban Movie", "豆瓣电影"), "https://movie.douban.com/", "life"),
  link(t("CupFox", "茶杯狐"), "https://cupfox.app/", "life"),
  link(t("Bilibili", "哔哩哔哩"), "https://www.bilibili.com/", "life"),
  link(t("Yiso Drive Search", "易搜网盘搜索"), "https://yiso.fun/", "life"),
  link("Kanopy", "https://lib.kanopy.com/", "life"),
  link("Our World in Data", "https://ourworldindata.org/", "life"),
  link(t("NGA Community", "NGA 玩家社区"), "https://ngabbs.com/", "life"),
  link(t("DJI Forum", "DJI 大疆社区"), "https://bbs.dji.com/", "life"),
  link(t("Zhihu", "知乎"), "https://www.zhihu.com/", "life"),
];

export const navigatorUpdated = {
  en: "July 2026",
  zh: "2026 年 7 月",
};
