import React from "react";
import Link from "@docusaurus/Link";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import Heading from "@theme/Heading";
import Layout from "@theme/Layout";
import AltmetricBadge from "@site/src/components/AltmetricBadge";
import { cvContent, cvIdentity } from "../../data/cvData";
import styles from "./styles.module.css";

const researchAccents = ["blue", "violet", "amber", "cyan", "green"];

function SectionHeading({ eyebrow, title, description, id }) {
  return (
    <div className={styles.sectionHeading}>
      <span className={styles.sectionEyebrow}>{eyebrow}</span>
      <Heading as="h2" id={id} className={styles.sectionTitle}>
        {title}
      </Heading>
      {description ? (
        <p className={styles.sectionDescription}>{description}</p>
      ) : null}
    </div>
  );
}

function DetailList({ items }) {
  return (
    <dl className={styles.detailList}>
      {items.map((item) => (
        <div key={item.label}>
          <dt>{item.label}</dt>
          <dd>{item.value}</dd>
        </div>
      ))}
    </dl>
  );
}

function ExternalAction({ href, children, ariaLabel, primary = false }) {
  return (
    <Link
      className={`${styles.heroAction} ${
        primary ? styles.heroActionPrimary : ""
      }`}
      to={href}
      target={href.startsWith("http") ? "_blank" : undefined}
      rel={href.startsWith("http") ? "noopener noreferrer" : undefined}
      aria-label={ariaLabel}
    >
      <span>{children}</span>
      <span className={styles.actionArrow} aria-hidden="true">
        {href.startsWith("mailto:") ? "@" : "↗"}
      </span>
    </Link>
  );
}

function ContactCard({ mark, label, hint, value, href, featured = false }) {
  const cardClassName = `${styles.contactCard} ${
    featured ? styles.contactCardFeatured : ""
  }`;

  const content = (
    <>
      <span className={styles.contactMark} aria-hidden="true">
        {mark}
      </span>
      <span className={styles.contactCopy}>
        <span className={styles.contactLabel}>{label}</span>
        <strong>{value}</strong>
        <span className={styles.contactHint}>{hint}</span>
      </span>
      {href ? (
        <span className={styles.contactArrow} aria-hidden="true">
          ↗
        </span>
      ) : null}
    </>
  );

  return href ? (
    <Link
      className={cardClassName}
      to={href}
      target={href.startsWith("http") ? "_blank" : undefined}
      rel={href.startsWith("http") ? "noopener noreferrer" : undefined}
    >
      {content}
    </Link>
  ) : (
    <div className={cardClassName}>{content}</div>
  );
}

function TimelineCard({ entry, showProjects = false }) {
  return (
    <article className={styles.timelineCard}>
      <div className={styles.timelineTopline}>
        <span className={styles.datePill}>{entry.date}</span>
      </div>
      <Heading as="h3" className={styles.cardTitle}>
        {entry.degree || entry.role}
      </Heading>
      <p className={styles.cardSubtitle}>{entry.institution}</p>
      {entry.description ? (
        <p className={styles.cardDescription}>{entry.description}</p>
      ) : null}
      {entry.details ? <DetailList items={entry.details} /> : null}
      {showProjects && entry.projects ? (
        <ul className={styles.projectList}>
          {entry.projects.map((project) => (
            <li key={project}>{project}</li>
          ))}
        </ul>
      ) : null}
    </article>
  );
}

function SkillGroup({ group }) {
  return (
    <article className={styles.skillCard}>
      <div className={styles.skillHeader}>
        <span className={styles.skillMark} aria-hidden="true">
          {group.mark}
        </span>
        <Heading as="h3">{group.title}</Heading>
      </div>
      <ul className={styles.skillList}>
        {group.items.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
    </article>
  );
}

export default function CurriculumVitaePage() {
  const { i18n } = useDocusaurusContext();
  const isChinese = i18n.currentLocale === "zh-Hans";
  const copy = isChinese ? cvContent.zh : cvContent.en;

  const contacts = [
    {
      mark: "@",
      label: copy.contacts.academic,
      hint: copy.contacts.academicHint,
      value: cvIdentity.academicEmail,
      href: `mailto:${cvIdentity.academicEmail}`,
      featured: true,
    },
    {
      mark: "BI",
      label: copy.contacts.business,
      hint: copy.contacts.businessHint,
      value: cvIdentity.businessEmail,
      href: `mailto:${cvIdentity.businessEmail}`,
    },
    {
      mark: "WX",
      label: copy.contacts.assistant,
      hint: copy.contacts.assistantHint,
      value: isChinese ? "微信咨询机器人 ↗" : "WeChat Support Bot ↗",
      href: "https://work.weixin.qq.com/kfid/kfc63941027aeefc636",
    },
    {
      mark: "SZ",
      label: copy.contacts.location,
      hint: copy.contacts.locationHint,
      value: copy.contacts.locationValue,
    },
    {
      mark: "WEB",
      label: copy.contacts.website,
      hint: copy.contacts.websiteHint,
      value: "smiler488.github.io",
      href: cvIdentity.website,
    },
  ];

  return (
    <Layout title={copy.meta.title} description={copy.meta.description}>
      <main className={styles.page}>
        <div className={styles.ambientOne} aria-hidden="true" />
        <div className={styles.ambientTwo} aria-hidden="true" />

        <div className={styles.shell}>
          <header className={styles.hero}>
            <div className={styles.heroIdentity}>
              <div className={styles.avatarFrame}>
                <img
                  src="/img/cv_person.png"
                  alt={copy.hero.name}
                  className={styles.avatar}
                  width="168"
                  height="168"
                />
                <span className={styles.statusDot} aria-hidden="true" />
              </div>

              <div className={styles.heroCopy}>
                <span className={styles.heroEyebrow}>{copy.hero.eyebrow}</span>
                <Heading as="h1" className={styles.heroTitle}>
                  {copy.hero.name}
                  <span>{copy.hero.secondaryName}</span>
                </Heading>
                <p className={styles.heroRole}>{copy.hero.role}</p>
                <p className={styles.heroInstitution}>
                  {copy.hero.institution}
                </p>
                <p className={styles.heroSummary}>{copy.hero.summary}</p>

                <div className={styles.heroPills}>
                  <span>{copy.hero.degree}</span>
                  <span>{copy.hero.appointment}</span>
                </div>
              </div>
            </div>

            <div className={styles.heroUtility}>
              <dl className={styles.heroStats}>
                {copy.hero.stats.map((stat) => (
                  <div key={stat.label}>
                    <dt>{stat.label}</dt>
                    <dd>{stat.value}</dd>
                  </div>
                ))}
              </dl>

              <div className={styles.heroActions}>
                <ExternalAction
                  href={`mailto:${cvIdentity.academicEmail}`}
                  primary
                >
                  {copy.actions.academic}
                </ExternalAction>
                <ExternalAction
                  href={cvIdentity.scholar}
                  ariaLabel={`${copy.actions.scholar} (${copy.actions.opensNewTab})`}
                >
                  {copy.actions.scholar}
                </ExternalAction>
                <ExternalAction
                  href={cvIdentity.orcid}
                  ariaLabel={`${copy.actions.orcid} (${copy.actions.opensNewTab})`}
                >
                  {copy.actions.orcid}
                </ExternalAction>
                <ExternalAction
                  href={cvIdentity.github}
                  ariaLabel={`${copy.actions.github} (${copy.actions.opensNewTab})`}
                >
                  {copy.actions.github}
                </ExternalAction>
                <button
                  type="button"
                  className={`${styles.heroAction} ${styles.printButton}`}
                  onClick={() => window.print()}
                >
                  <span>{copy.actions.print}</span>
                  <span className={styles.actionArrow} aria-hidden="true">
                    ⌘
                  </span>
                </button>
              </div>
            </div>
          </header>

          <section
            className={styles.contactSection}
            aria-labelledby="contact-channels"
          >
            <div className={styles.contactHeading}>
              <div>
                <span className={styles.sectionEyebrow}>
                  {copy.contacts.eyebrow}
                </span>
                <Heading
                  as="h2"
                  id="contact-channels"
                  className={styles.contactTitle}
                >
                  {copy.contacts.title}
                </Heading>
              </div>
              <p>{copy.contacts.description}</p>
            </div>

            <div className={styles.contactGrid}>
              {contacts.map((contact) => (
                <ContactCard key={contact.label} {...contact} />
              ))}
            </div>
          </section>

          <nav className={styles.sectionNav} aria-label={copy.navigation.label}>
            {copy.navigation.items.map((item, index) => (
              <Link key={item.href} to={item.href}>
                <span aria-hidden="true">
                  {String(index + 1).padStart(2, "0")}
                </span>
                {item.label}
              </Link>
            ))}
          </nav>

          <div className={styles.primaryGrid}>
            <div className={styles.primaryColumn}>
              <section aria-labelledby="research-focus">
                <SectionHeading
                  id="research-focus"
                  eyebrow={copy.research.eyebrow}
                  title={copy.research.title}
                  description={copy.research.description}
                />
                <div className={styles.researchGrid}>
                  {copy.research.items.map((item, index) => (
                    <article
                      key={item.title}
                      className={`${styles.researchCard} ${
                        styles[`accent_${researchAccents[index]}`]
                      }`}
                    >
                      <span className={styles.researchMark} aria-hidden="true">
                        {item.mark}
                      </span>
                      <Heading as="h3">{item.title}</Heading>
                      <p>{item.description}</p>
                    </article>
                  ))}
                </div>
              </section>

              <section aria-labelledby="appointment">
                <SectionHeading
                  id="appointment"
                  eyebrow={copy.appointment.eyebrow}
                  title={copy.appointment.title}
                />
                <TimelineCard entry={copy.appointment} />
              </section>

              <section aria-labelledby="education">
                <SectionHeading
                  id="education"
                  eyebrow={copy.education.eyebrow}
                  title={copy.education.title}
                />
                <div className={styles.educationGrid}>
                  {copy.education.entries.map((entry, index) => (
                    <TimelineCard
                      key={entry.degree}
                      entry={entry}
                      showProjects={index === 0}
                    />
                  ))}
                </div>
              </section>
            </div>

            <aside
              className={styles.skillsColumn}
              aria-labelledby="technical-skills"
            >
              <div className={styles.skillsPanel}>
                <SectionHeading
                  id="technical-skills"
                  eyebrow={copy.skills.eyebrow}
                  title={copy.skills.title}
                  description={copy.skills.description}
                />
                <div className={styles.skillGrid}>
                  {copy.skills.groups.map((group) => (
                    <SkillGroup key={group.title} group={group} />
                  ))}
                </div>
                <div className={styles.languageCard}>
                  <Heading as="h3">{copy.skills.languagesTitle}</Heading>
                  <ul>
                    {copy.skills.languages.map((language) => (
                      <li key={language}>{language}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </aside>
          </div>

          <section
            className={styles.fullSection}
            aria-labelledby="research-experience"
          >
            <SectionHeading
              id="research-experience"
              eyebrow={copy.experience.eyebrow}
              title={copy.experience.title}
              description={copy.experience.description}
            />
            <div className={styles.experienceGrid}>
              {copy.experience.entries.map((entry) => (
                <article key={entry.title} className={styles.experienceCard}>
                  <span className={styles.datePill}>{entry.date}</span>
                  <Heading as="h3">{entry.title}</Heading>
                  <p className={styles.cardSubtitle}>{entry.subtitle}</p>
                  <DetailList items={entry.details} />
                </article>
              ))}
            </div>
          </section>

          <section
            className={styles.fullSection}
            aria-labelledby="research-outputs"
          >
            <SectionHeading
              id="research-outputs"
              eyebrow={copy.outputs.eyebrow}
              title={copy.outputs.title}
              description={copy.outputs.description}
            />
            <div className={styles.outputGrid}>
              {copy.outputs.items.map((output) => (
                <article key={output.doi} className={styles.outputCard}>
                  <div className={styles.outputTopline}>
                    <span className={styles.outputMark} aria-hidden="true">
                      {output.mark}
                    </span>
                    <div>
                      <span>{output.kind}</span>
                      <strong>{output.year}</strong>
                    </div>
                  </div>
                  <Heading as="h3">{output.title}</Heading>
                  <p className={styles.outputVenue}>{output.venue}</p>
                  <p className={styles.outputAuthors}>{output.authors}</p>
                  {output.description ? (
                    <p className={styles.outputDescription}>
                      {output.description}
                    </p>
                  ) : null}
                  <div className={styles.outputFooter}>
                    <Link
                      to={output.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      aria-label={`DOI ${output.doi} (${copy.actions.opensNewTab})`}
                    >
                      DOI · {output.doi}
                      <span aria-hidden="true">↗</span>
                    </Link>
                    {output.doi === "10.1016/j.plaphe.2025.100135" ? (
                      <div className={styles.altmetricWrap}>
                        <AltmetricBadge doi={output.doi} badgeType="donut" />
                      </div>
                    ) : null}
                  </div>
                </article>
              ))}
            </div>
          </section>

          <footer className={styles.cvFooter}>
            <span>{copy.footer.updated}</span>
            <p>{copy.footer.note}</p>
          </footer>
        </div>
      </main>
    </Layout>
  );
}
