export const cvIdentity = {
  academicEmail: "googalphdlc@gmail.com",
  businessEmail: "dengliangchao@azureaxion.com",
  assistantEmail: "liangchaodeng@agent.qq.com",
  website: "https://smiler488.github.io/",
  scholar:
    "https://scholar.google.com/citations?hl=en&user=u3GFRMQAAAAJ&view_op=list_works",
  orcid: "https://orcid.org/0000-0002-5194-0655",
  github: "https://github.com/smiler488",
};

export const cvContent = {
  en: {
    meta: {
      title: "Curriculum Vitae",
      description:
        "Curriculum vitae of Liangchao Deng, a postdoctoral researcher working across plant phenotyping, computer vision, remote sensing, and crop modeling.",
    },
    hero: {
      eyebrow: "Curriculum vitae · Updated July 2026",
      name: "Liangchao Deng",
      secondaryName: "邓良超",
      role: "Postdoctoral Researcher",
      institution:
        "Shenzhen Institute of China Agricultural University · Shenzhen, China",
      summary:
        "Toward high, stable, and resource-efficient crop production under climate change, my research turns satellite–UAV–ground 3D observations into interpretable crop physiological states and, by coupling structure–radiation–photosynthesis–growth processes with data assimilation, builds verifiable, predictive, and intervenable crop digital twins that provide a mechanistic basis for the coordinated design of genotype × environment × management (G×E×M).",
      degree: "Ph.D. in Crop Science · Shihezi University · 2026",
      appointment: "Appointment · 1 Aug 2026 – 31 Jul 2029",
      stats: [
        { value: "5", label: "research themes" },
        { value: "2026", label: "Ph.D. awarded" },
        { value: "2026–29", label: "postdoctoral term" },
      ],
    },
    actions: {
      academic: "Academic email",
      scholar: "Google Scholar",
      orcid: "ORCID",
      github: "GitHub",
      print: "Print CV",
      opensNewTab: "opens in a new tab",
    },
    contacts: {
      eyebrow: "Contact channels",
      title: "Choose the right way to get in touch.",
      description:
        "Academic correspondence, commercial projects, and administrative communication are kept separate for faster replies.",
      academic: "Academic email",
      academicHint: "Research & collaboration",
      business: "Business inquiries",
      businessHint: "Commercial cooperation",
      assistant: "Assistant email",
      assistantHint: "Scheduling & coordination",
      location: "Location",
      locationValue: "Shenzhen, China",
      locationHint: "Postdoctoral appointment",
      website: "Website",
      websiteHint: "Projects & research updates",
    },
    navigation: {
      label: "CV sections",
      items: [
        { href: "#research-focus", label: "Research focus" },
        { href: "#appointment", label: "Appointment" },
        { href: "#education", label: "Education" },
        { href: "#technical-skills", label: "Skills" },
        { href: "#research-experience", label: "Experience" },
        { href: "#research-outputs", label: "Outputs" },
      ],
    },
    research: {
      eyebrow: "Research profile",
      title: "Five connected research themes",
      description:
        "A research program connecting field sensing, scientific computing, and deployable agricultural systems.",
      items: [
        {
          mark: "3D",
          title: "Agricultural Robotics & Computer Vision",
          description:
            "3D plant reconstruction, point-cloud structural analysis, robotic sensing, and autonomous data collection for high-throughput phenotyping.",
        },
        {
          mark: "AI",
          title: "AI-driven Plant Phenotyping",
          description:
            "Multimodal AI for image segmentation, object detection, phenotypic analysis, and structure–function modeling.",
        },
        {
          mark: "PAR",
          title: "Canopy Light & Photosynthesis",
          description:
            "Physics-based canopy light interception and photosynthesis modeling for efficient crop research.",
        },
        {
          mark: "UAV",
          title: "Remote Sensing & Sensor Fusion",
          description:
            "RGB, multispectral, hyperspectral, and LiDAR data fusion for smart agriculture.",
        },
        {
          mark: "DT",
          title: "Digital Twins & Process Models",
          description:
            "Coupling sensing data with crop growth models to build agricultural automation and simulation systems.",
        },
      ],
    },
    appointment: {
      eyebrow: "Current appointment",
      title: "Professional appointment",
      date: "1 August 2026 – 31 July 2029",
      role: "Postdoctoral Researcher",
      institution:
        "Shenzhen Institute of China Agricultural University, Shenzhen, China",
      description:
        "Postdoctoral research at the Shenzhen Institute of China Agricultural University.",
    },
    education: {
      eyebrow: "Academic training",
      title: "Education",
      entries: [
        {
          date: "2021 – 2026",
          degree: "Ph.D. in Crop Science (Integrated Master–Ph.D.)",
          institution: "Shihezi University, China",
          details: [
            {
              label: "Supervisors",
              value: "Prof. Yali Zhang; Dr. Qingfeng Song; Prof. Xin-Guang Zhu",
            },
            {
              label: "Research focus",
              value:
                "Crop phenomics, UAV remote sensing, canopy photosynthesis modeling, and AI-assisted phenotyping",
            },
            {
              label: "Joint training",
              value:
                "CAS Center for Excellence in Molecular Plant Sciences (CEMPS)",
            },
          ],
          projects: [
            "3D crop canopy reconstruction and light-distribution simulation using SfM and 3D Gaussian Splatting.",
            "Fusion of RGB, multispectral, hyperspectral, and LiDAR observations.",
            "Deep learning for segmentation, detection, and phenotypic-parameter prediction.",
          ],
        },
        {
          date: "2016 – 2021",
          degree: "B.Sc. in Information and Computational Science",
          institution: "Shihezi University, China",
          details: [
            {
              label: "Foundation",
              value:
                "Numerical analysis, computational modeling, programming, and algorithm design",
            },
            {
              label: "Core courses",
              value:
                "Computer vision, machine learning, linear algebra, optimization, graph theory, and data structures",
            },
            {
              label: "Graduation project",
              value:
                "Computational-fluid-dynamics simulation (Excellent Graduation Project)",
            },
          ],
        },
      ],
    },
    skills: {
      eyebrow: "Technical toolkit",
      title: "Technical skills",
      description:
        "Methods and tools used to move from sensing and modeling to reproducible research software.",
      groups: [
        {
          mark: "PY",
          title: "Programming & Data",
          items: [
            "Python · NumPy · SciPy",
            "PyTorch · OpenCV",
            "MATLAB · R",
            "AI product architecture",
          ],
        },
        {
          mark: "3D",
          title: "3D Vision & Point Clouds",
          items: [
            "SfM & photogrammetry",
            "PCL · Open3D",
            "Camera calibration",
            "Binocular vision",
          ],
        },
        {
          mark: "RS",
          title: "Remote Sensing",
          items: [
            "UAV imaging",
            "Multispectral & hyperspectral",
            "LiDAR processing",
            "Multi-source fusion",
          ],
        },
        {
          mark: "AI",
          title: "Machine Learning",
          items: [
            "Deep learning",
            "Phenotypic prediction",
            "Statistical modeling",
            "AI agent workflows",
          ],
        },
        {
          mark: "SIM",
          title: "Modeling & Simulation",
          items: [
            "Ray tracing & BRDF",
            "Photosynthesis simulation",
            "Digital-twin frameworks",
          ],
        },
        {
          mark: "DEV",
          title: "Research Software",
          items: [
            "Full-stack development",
            "Git version control",
            "Algorithm modularization",
          ],
        },
      ],
      languagesTitle: "Languages",
      languages: [
        "Chinese · Native",
        "English · Academic writing & scientific communication",
      ],
    },
    experience: {
      eyebrow: "Selected work",
      title: "Research experience",
      description:
        "Selected programs showing the progression from sensing and reconstruction to simulation and applied phenotyping.",
      entries: [
        {
          date: "2023 – Present",
          title: "AI-assisted 3D Crop Canopy Modeling",
          subtitle: "3D reconstruction, light distribution & photosynthesis",
          details: [
            {
              label: "3D canopy reconstruction",
              value:
                "High-throughput farmland reconstruction using SfM, 3D Gaussian Splatting, and UAV cross-circular acquisition, reaching centimeter-level accuracy.",
            },
            {
              label: "Light & photosynthesis",
              value:
                "Canopy-scale simulation using ray tracing and BRDF-based leaf optics within a crop digital-twin framework.",
            },
            {
              label: "Multimodal AI",
              value:
                "RGB, multispectral, and LiDAR workflows for complex-scene, zero-shot plant segmentation.",
            },
            {
              label: "Modular research agent",
              value:
                "Integrated reconstruction, meshing, canopy generation, light simulation, and photosynthesis calculation into reusable modules.",
            },
          ],
        },
        {
          date: "2021 – 2023",
          title: "High-throughput 3D & Spectral Phenotyping",
          subtitle: "Optical inversion, crop design & computer vision",
          details: [
            {
              label: "Leaf optical inversion",
              value:
                "Developed a BRDF-based inversion framework and optimized measurement schemes for indirect leaf-optics estimation.",
            },
            {
              label: "Wheat design research",
              value:
                "Modeled plant architecture and cultivation configurations for higher canopy efficiency, including industry-oriented collaboration.",
            },
            {
              label: "Phenotyping algorithms",
              value:
                "Built computer-vision methods for automated extraction and analysis of plant phenotypic traits.",
            },
          ],
        },
      ],
    },
    outputs: {
      eyebrow: "Research outputs",
      title: "Publications & software",
      description:
        "Peer-reviewed research and reusable software supporting plant phenotyping workflows.",
      items: [
        {
          mark: "PP",
          kind: "Peer-reviewed article",
          year: "2025",
          title:
            "Leaf Bidirectional Reflectance Distribution Function (BRDF) Prediction with Phenotypic Traits in Four Species: Development of a Novel Measuring and Analyzing Framework",
          venue: "Plant Phenomics",
          authors:
            "Deng, L.; Yu, L. X.; Mao, L.; Wang, Y.; Guo, X.; Wang, M.; Zhang, Y.; Song, Q.; Zhu, X.-G.",
          doi: "10.1016/j.plaphe.2025.100135",
          url: "https://doi.org/10.1016/j.plaphe.2025.100135",
        },
        {
          mark: "SW",
          kind: "Research software",
          year: "2025",
          title: "Digital Plant Phenotyping Platform (v25.0)",
          venue: "Zenodo",
          authors: "Deng, L.",
          doi: "10.5281/zenodo.17544584",
          url: "https://doi.org/10.5281/zenodo.17544584",
          description:
            "An integrated platform for plant phenotyping, data processing, and analysis. Core modules have been transferred through Shufeng Bio for applied phenotyping and intelligent-agriculture services.",
        },
      ],
    },
    footer: {
      updated: "Last updated · July 2026",
      note: "For academic correspondence, please use the dedicated academic email above.",
    },
  },
  zh: {
    meta: {
      title: "个人简历",
      description:
        "邓良超博士后研究人员的个人简历，研究方向涵盖植物表型、计算机视觉、遥感与作物模型。",
    },
    hero: {
      eyebrow: "个人简历 · 2026年7月更新",
      name: "邓良超",
      secondaryName: "Liangchao Deng",
      role: "博士后研究人员",
      institution:
        "深圳市中农大前沿技术研究院（中国农业大学深圳研究院）· 中国深圳",
      summary:
        "面向气候变化下的作物高产、稳产与资源高效，研究如何将卫星—无人机—地面三维观测转化为可解释的作物生理状态，并通过结构—辐射—光合—生长过程耦合和数据同化，构建可验证、可推演、可干预的作物数字孪生，为品种×环境×管理协同设计提供机制依据。",
      degree: "作物科学博士 · 石河子大学 · 2026年获授",
      appointment: "聘期 · 2026年8月1日 – 2029年7月31日",
      stats: [
        { value: "5", label: "个研究主题" },
        { value: "2026", label: "博士学位获授" },
        { value: "2026–29", label: "博士后聘期" },
      ],
    },
    actions: {
      academic: "学术邮件",
      scholar: "Google Scholar",
      orcid: "ORCID",
      github: "GitHub",
      print: "打印简历",
      opensNewTab: "将在新标签页打开",
    },
    contacts: {
      eyebrow: "联系方式",
      title: "根据事项选择合适的联系渠道。",
      description:
        "学术交流、商业合作与行政协调分别使用独立邮箱，便于更快处理与回复。",
      academic: "学术邮箱",
      academicHint: "科研交流与合作",
      business: "商业合作",
      businessHint: "项目与商业咨询",
      assistant: "助理邮箱",
      assistantHint: "日程与事务协调",
      location: "所在地",
      locationValue: "中国深圳",
      locationHint: "博士后工作地点",
      website: "个人网站",
      websiteHint: "项目与研究动态",
    },
    navigation: {
      label: "简历章节",
      items: [
        { href: "#research-focus", label: "研究方向" },
        { href: "#appointment", label: "任职经历" },
        { href: "#education", label: "教育经历" },
        { href: "#technical-skills", label: "技术技能" },
        { href: "#research-experience", label: "研究经历" },
        { href: "#research-outputs", label: "研究成果" },
      ],
    },
    research: {
      eyebrow: "研究概览",
      title: "五个相互衔接的研究主题",
      description:
        "以田间感知、科学计算和可落地农业系统为主线构建完整研究体系。",
      items: [
        {
          mark: "3D",
          title: "农业机器人与计算机视觉",
          description:
            "3D 植物重建、点云结构分析，以及面向高通量表型的机器人感知与自主数据采集。",
        },
        {
          mark: "AI",
          title: "AI 驱动的植物表型分析",
          description:
            "多模态 AI 在图像分割、目标检测、植物表型分析与结构—功能建模中的应用。",
        },
        {
          mark: "PAR",
          title: "冠层光截获与光合作用",
          description: "面向高效作物研究的物理机理冠层光截获与光合作用建模。",
        },
        {
          mark: "UAV",
          title: "遥感与多源传感器融合",
          description:
            "融合 RGB、多光谱、高光谱和 LiDAR 数据，服务智慧农业研究。",
        },
        {
          mark: "DT",
          title: "数字孪生与过程模型",
          description:
            "将感知数据与作物生长模型耦合，构建农业自动化与仿真系统。",
        },
      ],
    },
    appointment: {
      eyebrow: "当前任职",
      title: "任职经历",
      date: "2026年8月1日 – 2029年7月31日",
      role: "博士后研究人员",
      institution:
        "深圳市中农大前沿技术研究院（中国农业大学深圳研究院），中国深圳",
      description: "在深圳市中农大前沿技术研究院从事博士后研究。",
    },
    education: {
      eyebrow: "学术训练",
      title: "教育经历",
      entries: [
        {
          date: "2021 – 2026",
          degree: "作物科学博士（硕博连读）",
          institution: "石河子大学，中国",
          details: [
            {
              label: "导师",
              value: "张亚莉教授；宋庆峰博士；朱新广教授",
            },
            {
              label: "研究方向",
              value:
                "作物表型组学、无人机遥感、冠层光合作用建模与 AI 辅助表型分析",
            },
            {
              label: "联合培养",
              value: "中国科学院分子植物科学卓越创新中心（CEMPS）",
            },
          ],
          projects: [
            "基于 SfM 与 3D 高斯溅射的作物冠层重建与光分布模拟。",
            "RGB、多光谱、高光谱和 LiDAR 观测数据的协同融合。",
            "用于分割、检测和表型参数预测的深度学习方法。",
          ],
        },
        {
          date: "2016 – 2021",
          degree: "信息与计算科学学士",
          institution: "石河子大学，中国",
          details: [
            {
              label: "专业基础",
              value: "数值分析、计算建模、程序设计与算法设计",
            },
            {
              label: "核心课程",
              value: "计算机视觉、机器学习、线性代数、优化算法、图论和数据结构",
            },
            {
              label: "毕业设计",
              value: "基于计算流体动力学的数值模拟（优秀毕业设计）",
            },
          ],
        },
      ],
    },
    skills: {
      eyebrow: "技术工具箱",
      title: "技术技能",
      description: "覆盖感知、建模、分析与可复现科研软件开发的方法和工具。",
      groups: [
        {
          mark: "PY",
          title: "编程与数据分析",
          items: [
            "Python · NumPy · SciPy",
            "PyTorch · OpenCV",
            "MATLAB · R",
            "AI 产品架构设计",
          ],
        },
        {
          mark: "3D",
          title: "三维视觉与点云",
          items: ["SfM 与摄影测量", "PCL · Open3D", "相机标定", "双目视觉"],
        },
        {
          mark: "RS",
          title: "遥感与传感",
          items: [
            "无人机成像",
            "多光谱与高光谱",
            "LiDAR 数据处理",
            "多源传感器融合",
          ],
        },
        {
          mark: "AI",
          title: "机器学习",
          items: ["深度学习", "表型预测", "统计建模", "AI 智能体工作流"],
        },
        {
          mark: "SIM",
          title: "建模与仿真",
          items: ["光线追踪与 BRDF", "光合作用模拟", "数字孪生框架"],
        },
        {
          mark: "DEV",
          title: "科研软件",
          items: ["全栈开发", "Git 版本控制", "算法模块化"],
        },
      ],
      languagesTitle: "语言能力",
      languages: ["中文 · 母语", "英语 · 学术写作与科学交流"],
    },
    experience: {
      eyebrow: "代表性工作",
      title: "研究经历",
      description: "从感知和三维重建逐步拓展至模拟计算与应用型植物表型研究。",
      entries: [
        {
          date: "2023 – 至今",
          title: "AI 辅助的 3D 作物冠层建模",
          subtitle: "三维重建、光分布与光合作用模拟",
          details: [
            {
              label: "冠层三维重建",
              value:
                "结合 SfM、3D 高斯溅射与无人机交叉环形采集，实现厘米级精度的农田高通量重建。",
            },
            {
              label: "光分布与光合作用",
              value:
                "将光线追踪和基于 BRDF 的叶片光学特性集成至作物数字孪生框架。",
            },
            {
              label: "多模态 AI",
              value:
                "构建 RGB、多光谱和 LiDAR 工作流，实现复杂场景下的零样本植物分割。",
            },
            {
              label: "模块化科研智能体",
              value:
                "将重建、网格化、冠层生成、光模拟与光合作用计算整合为可复用模块。",
            },
          ],
        },
        {
          date: "2021 – 2023",
          title: "高通量 3D 与光谱表型分析",
          subtitle: "光学反演、作物设计与计算机视觉",
          details: [
            {
              label: "叶片光学反演",
              value:
                "开发基于 BRDF 的反演框架并优化测量方案，实现叶片光学特性的间接估计。",
            },
            {
              label: "小麦设计研究",
              value:
                "围绕更高冠层效率开展株型与栽培配置建模，包括面向产业的协作研究。",
            },
            {
              label: "表型算法",
              value: "构建用于植物表型性状自动提取和分析的计算机视觉方法。",
            },
          ],
        },
      ],
    },
    outputs: {
      eyebrow: "研究成果",
      title: "论文与软件",
      description: "服务植物表型工作流的同行评议研究与可复用科研软件。",
      items: [
        {
          mark: "PP",
          kind: "同行评议论文",
          year: "2025",
          title:
            "Leaf Bidirectional Reflectance Distribution Function (BRDF) Prediction with Phenotypic Traits in Four Species: Development of a Novel Measuring and Analyzing Framework",
          venue: "Plant Phenomics",
          authors:
            "Deng, L.; Yu, L. X.; Mao, L.; Wang, Y.; Guo, X.; Wang, M.; Zhang, Y.; Song, Q.; Zhu, X.-G.",
          doi: "10.1016/j.plaphe.2025.100135",
          url: "https://doi.org/10.1016/j.plaphe.2025.100135",
        },
        {
          mark: "SW",
          kind: "科研软件",
          year: "2025",
          title: "Digital Plant Phenotyping Platform (v25.0)",
          venue: "Zenodo",
          authors: "Deng, L.",
          doi: "10.5281/zenodo.17544584",
          url: "https://doi.org/10.5281/zenodo.17544584",
          description:
            "集成植物表型分析、数据处理与分析的软件平台。核心模块已通过舒丰生物完成转让和商业化，用于植物表型与智慧农业服务。",
        },
      ],
    },
    footer: {
      updated: "最后更新 · 2026年7月",
      note: "学术交流请优先使用上方独立学术邮箱。",
    },
  },
};
