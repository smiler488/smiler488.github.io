const link = (title, url, category) => ({ title, url, category });

export const navigatorCategories = [
  {
    id: "ai-science",
    label: { en: "AI for Science", zh: "AI for Science" },
    description: {
      en: "Scientific AI platforms, literature intelligence, datasets, and reproducible benchmarks.",
      zh: "科研智能平台、文献发现、开放数据集与可复现实验基准。",
    },
    keywords: ["science", "papers", "datasets", "科研", "文献", "数据集"],
  },
  {
    id: "digital-crops",
    label: {
      en: "Crop models & digital plants",
      zh: "作物模型与数字植株",
    },
    description: {
      en: "Functional-structural plant models, crop simulation, ray tracing, and virtual plants.",
      zh: "功能结构植物模型、作物模拟、光线追踪与虚拟植物。",
    },
    keywords: ["crop", "simulation", "FSPM", "作物", "模型", "虚拟植物"],
  },
  {
    id: "phenotyping",
    label: { en: "Phenotyping & standards", zh: "植物表型与标准" },
    description: {
      en: "Phenotyping networks, open datasets, breeding APIs, and metadata standards.",
      zh: "表型网络、开放数据集、育种接口与元数据标准。",
    },
    keywords: ["phenotyping", "breeding", "BrAPI", "表型", "育种", "标准"],
  },
  {
    id: "physiology",
    label: { en: "Plant physiology & measurement", zh: "植物生理与测量" },
    description: {
      en: "Photosynthesis, fluorescence, canopy sensing, protocols, and analysis references.",
      zh: "光合作用、荧光、冠层测量、实验协议与数据分析参考。",
    },
    keywords: [
      "photosynthesis",
      "fluorescence",
      "instruments",
      "光合",
      "荧光",
      "仪器",
    ],
  },
  {
    id: "geo-remote",
    label: { en: "Remote sensing & 3D", zh: "遥感、地理与三维" },
    description: {
      en: "Earth observation, geospatial data, terrain, photogrammetry, and point-cloud workflows.",
      zh: "地球观测、空间数据、地形、摄影测量与点云工作流。",
    },
    keywords: ["remote sensing", "GIS", "UAV", "3D", "遥感", "无人机", "点云"],
  },
  {
    id: "data-orgs",
    label: {
      en: "Agricultural data & organizations",
      zh: "农业科研数据与机构",
    },
    description: {
      en: "Trusted agricultural, climate, soil, and institutional data gateways.",
      zh: "农业、气候、土壤及权威科研机构的数据入口。",
    },
    keywords: ["data", "climate", "agriculture", "数据", "气候", "农业机构"],
  },
  {
    id: "ai-ecosystem",
    label: { en: "AI models & agents", zh: "AI 模型与智能体" },
    description: {
      en: "Model hubs, developer platforms, local inference, and open-source serving tools.",
      zh: "模型社区、开发平台、本地推理与开源模型服务工具。",
    },
    keywords: ["AI", "LLM", "models", "模型", "推理", "智能体"],
  },
  {
    id: "writing-viz",
    label: {
      en: "Scientific writing & visualization",
      zh: "论文写作与科研可视化",
    },
    description: {
      en: "Core journals, scientific writing aids, terminology, plots, and diagram tools.",
      zh: "核心期刊、科研写作、术语检索、统计绘图与图表工具。",
    },
    keywords: ["journal", "writing", "visualization", "期刊", "写作", "绘图"],
  },
  {
    id: "learning-code",
    label: { en: "Courses, code & open tools", zh: "课程、代码与开放工具" },
    description: {
      en: "Computer vision, machine learning, statistics, scientific software, and open courses.",
      zh: "计算机视觉、机器学习、统计方法、科研软件与开放课程。",
    },
    keywords: ["course", "code", "statistics", "课程", "代码", "统计"],
  },
];

export const navigatorLinks = [
  link("Hugging Face", "https://huggingface.co/", "ai-ecosystem"),
  link("ModelScope", "https://www.modelscope.cn/home", "ai-ecosystem"),
  link("OpenXLab", "https://openxlab.org.cn/apps", "ai-ecosystem"),
  link("Weights & Biases", "https://wandb.ai/site", "ai-ecosystem"),
  link("Ollama", "https://ollama.com/", "ai-ecosystem"),
  link(
    "OpenAI Codex CLI Docs",
    "https://developers.openai.com/codex/cli",
    "ai-ecosystem"
  ),
  link("Anthropic", "https://www.anthropic.com/", "ai-ecosystem"),
  link("Google AI Studio", "https://aistudio.google.com/", "ai-ecosystem"),
  link("OpenRouter", "https://openrouter.ai/", "ai-ecosystem"),
  link(
    "Transformers",
    "https://github.com/huggingface/transformers",
    "ai-ecosystem"
  ),
  link("vLLM", "https://github.com/vllm-project/vllm", "ai-ecosystem"),
  link("SGLang", "https://docs.sglang.io/", "ai-ecosystem"),
  link(
    "Ultralytics YOLO Docs",
    "https://docs.ultralytics.com/zh/",
    "ai-ecosystem"
  ),

  link("AI4S-YB", "https://ai4s-yb.org/", "ai-science"),
  link("ScienceOne", "https://scienceone.ia.ac.cn/", "ai-science"),
  link("SciSpace", "https://typeset.io/", "ai-science"),
  link("TXYZ", "https://www.txyz.ai/", "ai-science"),
  link("Semantic Scholar", "https://www.semanticscholar.org/", "ai-science"),
  link("Connected Papers", "https://www.connectedpapers.com/", "ai-science"),
  link("alphaXiv", "https://www.alphaxiv.org/", "ai-science"),
  link("bioRxiv", "https://www.biorxiv.org/", "ai-science"),
  link("Kaggle Datasets", "https://www.kaggle.com/datasets", "ai-science"),
  link("Codabench", "https://www.codabench.org/", "ai-science"),

  link("Crops in Silico", "https://cropsinsilico.org/", "digital-crops"),
  link(
    "GreenLab",
    "https://greenlab.ac.cn/index.php/%E9%A6%96%E9%A1%B5",
    "digital-crops"
  ),
  link(
    "GEMINI Breeding",
    "https://gemini-breeding.github.io/",
    "digital-crops"
  ),
  link("BioCro", "https://github.com/biocro/biocro", "digital-crops"),
  link(
    "OpenAlea Plant Biophysics",
    "https://openalea.readthedocs.io/en/latest/packages/index.html#plant-biophysics",
    "digital-crops"
  ),
  link(
    "Helios",
    "https://baileylab.ucdavis.edu/software/helios/index.html",
    "digital-crops"
  ),
  link("LESS", "https://lessrt.org/", "digital-crops"),
  link(
    "Plant Simulation Lab",
    "https://baileylab.ucdavis.edu/research/index.html",
    "digital-crops"
  ),
  link("RIPE", "https://ripe.illinois.edu/", "digital-crops"),
  link("Plant Moves", "https://plantmoves.nl/", "digital-crops"),

  link(
    "International Plant Phenotyping Network",
    "https://www.plant-phenotyping.org/",
    "phenotyping"
  ),
  link("EMPHASIS", "https://emphasis.plant-phenotyping.eu/", "phenotyping"),
  link("NPEC", "https://www.npec.nl/", "phenotyping"),
  link("PhenoRob", "https://www.phenorob.de/index.html", "phenotyping"),
  link(
    "ORNL Advanced Plant Phenotyping Lab",
    "https://www.ornl.gov/appl",
    "phenotyping"
  ),
  link(
    "Laboratory of Field Phenomics",
    "https://lab.fieldphenomics.com/index.html",
    "phenotyping"
  ),
  link("PhenoNet", "https://phenonet.org/", "phenotyping"),
  link(
    "Pheno4D Dataset",
    "https://www.ipb.uni-bonn.de/data/pheno4d/",
    "phenotyping"
  ),
  link("BrAPI", "https://brapi.org/", "phenotyping"),
  link("MIAPPE", "https://www.miappe.org/support/", "phenotyping"),
  link("BreedBase", "https://breedbase.org/", "phenotyping"),
  link(
    "Integrated Breeding Platform",
    "https://www.integratedbreeding.net/",
    "phenotyping"
  ),
  link(
    "NARO Rootomics",
    "https://www.naro.go.jp/phenotyping/rootomics_db/",
    "phenotyping"
  ),

  link(
    "Google Earth Engine Code Editor",
    "https://code.earthengine.google.com/",
    "geo-remote"
  ),
  link(
    "NASA POWER",
    "https://power.larc.nasa.gov/data-access-viewer/",
    "geo-remote"
  ),
  link("OpenTopography", "https://opentopography.org/", "geo-remote"),
  link(
    "Copernicus FAPAR",
    "https://land.copernicus.eu/global/products/fapar",
    "geo-remote"
  ),
  link("LP DAAC", "https://lpdaac.usgs.gov/product_search/", "geo-remote"),
  link("MODIS", "https://modis.gsfc.nasa.gov/data/", "geo-remote"),
  link("Alaska Satellite Facility", "https://asf.alaska.edu/", "geo-remote"),
  link(
    "QGIS User Guide",
    "https://docs.qgis.org/3.40/en/docs/user_manual/",
    "geo-remote"
  ),
  link("OpenDroneMap Docs", "https://docs.opendronemap.org/", "geo-remote"),
  link("RESDC", "https://www.resdc.cn/", "geo-remote"),
  link("OpenGMS", "https://geomodeling.njnu.edu.cn/", "geo-remote"),
  link("ECMWF", "https://www.ecmwf.int/", "geo-remote"),
  link(
    "LAADS DAAC",
    "https://ladsweb.modaps.eosdis.nasa.gov/search/",
    "geo-remote"
  ),

  link("FAOSTAT", "https://www.fao.org/faostat/en/#data/FBS", "data-orgs"),
  link(
    "Harmonized World Soil Database",
    "https://www.fao.org/soils-portal/soil-survey/soil-maps-and-databases/harmonized-world-soil-database-v12/en/",
    "data-orgs"
  ),
  link("NOAA", "https://www.noaa.gov/", "data-orgs"),
  link(
    "China Meteorological Data Service",
    "https://data.cma.cn/",
    "data-orgs"
  ),
  link(
    "National Agricultural Science Data Center",
    "https://www.agridata.cn/#/home",
    "data-orgs"
  ),
  link(
    "Agricultural Knowledge Service",
    "https://agri.nais.net.cn/index.html",
    "data-orgs"
  ),
  link(
    "International Rice Research Institute",
    "https://www.irri.org/",
    "data-orgs"
  ),
  link("CIMMYT", "https://www.cimmyt.org/about/", "data-orgs"),
  link("IPCC", "https://www.ipcc.ch/", "data-orgs"),
  link("National Academies", "https://www.nationalacademies.org/", "data-orgs"),
  link("ASABE", "https://asabe.org/About-Us", "data-orgs"),
  link("U.S. AgLab", "https://aglab.ars.usda.gov/", "data-orgs"),
  link(
    "International Cotton Advisory Committee",
    "https://icac.org/",
    "data-orgs"
  ),

  link(
    "LI-6800 Photosynthesis System",
    "https://www.licor.com/env/products/photosynthesis/LI-6800/",
    "physiology"
  ),
  link(
    "Photosynthesis Data Analysis with R",
    "https://bookdown.org/zhujiedong/photoanalysis/docs/",
    "physiology"
  ),
  link("PhotosynQ", "https://www.photosynq.com/", "physiology"),
  link(
    "PhotosynQ Documentation",
    "https://help.photosynq.com/#measurements",
    "physiology"
  ),
  link(
    "PROMETHEUS Protocols",
    "https://prometheusprotocols.net/",
    "physiology"
  ),
  link("Plants in Action", "https://rseco.org/index.html", "physiology"),
  link(
    "WUR Plant Physiology",
    "https://www.wur.nl/en/Research-Results/Chair-groups/Plant-Sciences/Laboratory-of-Plant-Physiology.htm",
    "physiology"
  ),
  link(
    "Leaf Area Index Guide",
    "https://www.metergroup.com/en/meter-environment/education-guides/researchers-complete-guide-leaf-area-index-lai",
    "physiology"
  ),
  link(
    "FloX SIF Monitoring",
    "https://www.jb-hyperspectral.com/products/flox/",
    "physiology"
  ),
  link("Stomata Overview", "https://stomata.uvm.edu/", "physiology"),

  link(
    "Manchester Academic Phrasebank",
    "https://www.phrasebank.manchester.ac.uk/",
    "writing-viz"
  ),
  link("Purdue OWL", "https://owl.purdue.edu/owl/index.html", "writing-viz"),
  link(
    "Overleaf Templates",
    "https://www.overleaf.com/latex/templates/?nocdn=true",
    "writing-viz"
  ),
  link("CAS Source Index", "https://cassi.cas.org/search.jsp", "writing-viz"),
  link("Planteome", "https://planteome.org/", "writing-viz"),
  link(
    "Annual Review of Plant Biology",
    "https://www.annualreviews.org/content/journals/arplant",
    "writing-viz"
  ),
  link(
    "Plant Phenomics",
    "https://spj.science.org/journal/plantphenomics",
    "writing-viz"
  ),
  link(
    "Plant Methods",
    "https://plantmethods.biomedcentral.com/",
    "writing-viz"
  ),
  link(
    "in silico Plants",
    "https://academic.oup.com/insilicoplants/",
    "writing-viz"
  ),
  link(
    "Journal of Experimental Botany",
    "https://academic.oup.com/jxb",
    "writing-viz"
  ),
  link(
    "Remote Sensing of Environment",
    "https://www.sciencedirect.com/journal/remote-sensing-of-environment",
    "writing-viz"
  ),
  link("The R Graph Gallery", "https://r-graph-gallery.com/", "writing-viz"),
  link(
    "Seaborn Example Gallery",
    "https://seaborn.pydata.org/examples/index.html",
    "writing-viz"
  ),
  link(
    "From Data to Viz",
    "https://www.data-to-viz.com/#density",
    "writing-viz"
  ),
  link(
    "Apache ECharts Examples",
    "https://echarts.apache.org/examples/zh/index.html",
    "writing-viz"
  ),
  link("diagrams.net", "https://app.diagrams.net/", "writing-viz"),

  link(
    "Stanford CS231n",
    "https://cs231n.stanford.edu/schedule.html",
    "learning-code"
  ),
  link(
    "Stanford CS109",
    "https://web.stanford.edu/class/cs109/",
    "learning-code"
  ),
  link("MIT OpenCourseWare", "https://ocw.mit.edu/", "learning-code"),
  link(
    "MIT Introduction to Machine Learning",
    "https://introml.mit.edu/notes/",
    "learning-code"
  ),
  link(
    "MIT Deep Learning Lectures",
    "https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/video_galleries/lecture-videos/",
    "learning-code"
  ),
  link("Stanford CS230", "https://cs230.stanford.edu/", "learning-code"),
  link(
    "Python NumPy Tutorial",
    "https://cs231n.github.io/python-numpy-tutorial/",
    "learning-code"
  ),
  link(
    "Dive into CV with PyTorch",
    "https://datawhalechina.github.io/dive-into-cv-pytorch/#/?id=dive-into-cv-pytorch",
    "learning-code"
  ),
  link(
    "Modern Statistical Graphics",
    "https://bookdown.org/xiangyun/msg/#welcome",
    "learning-code"
  ),
  link(
    "Statistical Analysis with R",
    "https://xueningzhu.github.io/Statistical-Analysis-with-R/index.html",
    "learning-code"
  ),
  link("NeRF", "https://github.com/bmild/nerf", "learning-code"),
  link(
    "Zotero Chinese Community",
    "https://zotero-chinese.com/",
    "learning-code"
  ),
  link(
    "Tsinghua Open Source Mirror",
    "https://mirrors.tuna.tsinghua.edu.cn/",
    "learning-code"
  ),
];

export const navigatorUpdated = {
  en: "July 2026",
  zh: "2026 年 7 月",
};
