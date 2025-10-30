import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: 'Digital Crop Phenotyping',
    Svg: require('@site/static/img/compress_comic1.svg').default,
    description: (
      <>
        Quantifying crop canopy and organ structures via multi-view 3D reconstruction and UAV imaging, enabling efficient, multi-scale phenotypic monitoring for structural and functional analysis.
      </>
    ),
  },
  {
    title: 'AI-powered Phenomic Analysis',
    Svg: require('@site/static/img/compress_comic2.svg').default,
    description: (
      <>
        Integrating computer vision and large language models to automate phenotypic data processing, enhance feature recognition, and construct intelligent analytical frameworks for crop science.
      </>
    ),
  },
  {
    title: 'Canopy Photosynthesis & Breeding',
    Svg: require('@site/static/img/compress_comic3.svg').default,
    description: (
      <>
        Linking photosynthesis models with phenotypic and environmental data to guide high-efficiency canopy design, intelligent breeding, and sustainable crop production strategies.
      </>
    ),
  },
];

function Feature({ Svg, title, description }) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
