import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: 'Crop Phenotyping',
    Svg: require('@site/static/img/comicpho1.svg').default,
    description: (
      <>
        Leveraging multi-view 3D point cloud reconstruction to quantify crop canopy and organ
        structures, enabling high-throughput phenotypic trait extraction for multi-scale monitoring
        and analysis.
      </>
    ),
  },
  {
    title: 'AI-driven Researching',
    Svg: require('@site/static/img/comicuav1.svg').default,
    description: (
      <>
        Applying computer vision, large language models, and artificial intelligence to automate
        phenotypic data analysis, enhance feature recognition, and build intelligent frameworks for
        crop research.
      </>
    ),
  },
  {
    title: 'Canopy Photosynthesis Modeling for Breeding & Production',
    Svg: require('@site/static/img/comicexp1.svg').default,
    description: (
      <>
        Integrating canopy photosynthesis models with breeding strategies and field management,
        guiding the strategy of high-photosynthetic-efficiency under changing environments.
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
