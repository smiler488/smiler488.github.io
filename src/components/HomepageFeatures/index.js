import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: 'Physiological and Ecological Field',
    Svg: require('@site/static/img/comicpho.svg').default,
    description: (
      <>
        Integrating environmental factors with canopy photosynthesis models to explore the regulatory mechanisms of plant architecture, cultivation practices, and environmental conditions on cotton canopy photosynthesis efficiency.
      </>
    ),
  },
  {
    title: 'Phenotyping Research Field',
    Svg: require('@site/static/img/comicuav.svg').default,
    description: (
      <>
        Utilizing 3D point cloud reconstruction technology to quantify plant organs and canopy structures, constructing novel high-throughput structural phenotypic metrics.
      </>
    ),
  },
  {
    title: 'Crop Design Field',
    Svg: require('@site/static/img/comicexp.svg').default,
    description: (
      <>
        Integrating genetic diversity with phenotypic data to optimize plant architecture and cultivation practices, cultivating high-photosynthetic-efficiency cotton to adapt to climate change.
      </>
    ),
  },
];

function Feature({Svg, title, description}) {
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
