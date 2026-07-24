import React from "react";
import Link from "@docusaurus/Link";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import Heading from "@theme/Heading";
import Layout from "@theme/Layout";
import AltmetricBadge from "@site/src/components/AltmetricBadge";
import { cvContent, cvIdentity } from "../../data/cvData";
import styles from "./styles.module.css";

/* ── SVG icon library for card mark badges ── */
const ICONS = {
  mail: (s) => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={s}>
      <rect x="2" y="4" width="20" height="16" rx="2" />
      <path d="m22 7-8.97 5.72a1.94 1.94 0 0 1-2.06 0L2 7" />
    </svg>
  ),
  briefcase: (s) => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={s}>
      <rect x="2" y="7" width="20" height="14" rx="2" ry="2" />
      <path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16" />
    </svg>
  ),
  calendar: (s) => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={s}>
      <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
      <line x1="16" y1="2" x2="16" y2="6" />
      <line x1="8" y1="2" x2="8" y2="6" />
      <line x1="3" y1="10" x2="21" y2="10" />
    </svg>
  ),
  pin: (s) => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={s}>
      <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z" />
      <circle cx="12" cy="10" r="3" />
    </svg>
  ),
  globe: (s) => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={s}>
      <circle cx="12" cy="12" r="10" />
      <line x1="2" y1="12" x2="22" y2="12" />
      <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
    </svg>
  ),
  cube: (s) => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={s}>
      <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
      <polyline points="3.27 6.96 12 12.01 20.73 6.96" />
      <line x1="12" y1="22.08" x2="12" y2="12" />
    </svg>
  ),
  brain: (s) => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={s}>
      <path d="M12 2a4 4 0 0 0-4 4v1a3 3 0 0 0-3 3 3 3 0 0 0 1.1 2.3A3.5 3.5 0 0 0 5 16a3.5 3.5 0 0 0 3.5 3.5h1V22h5v-2.5h1A3.5 3.5 0 0 0 19 16a3.5 3.5 0 0 0-1.1-3.7A3 3 0 0 0 19 10a3 3 0 0 0-3-3V6a4 4 0 0 0-4-4z" />
      <line x1="12" y1="2" x2="12" y2="22" />
    </svg>
  ),
  sun: (s) => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={s}>
      <circle cx="12" cy="12" r="5" />
      <line x1="12" y1="1" x2="12" y2="3" />
      <line x1="12" y1="21" x2="12" y2="23" />
      <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
      <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
      <line x1="1" y1="12" x2="3" y2="12" />
      <line x1="21" y1="12" x2="23" y2="12" />
      <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
      <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
    </svg>
  ),
  satellite: (s) => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={s}>
      <circle cx="12" cy="12" r="3" />
      <path d="M12 2v4" />
      <path d="M12 18v4" />
      <path d="M4.93 4.93l2.83 2.83" />
      <path d="M16.24 16.24l2.83 2.83" />
      <path d="M2 12h4" />
      <path d="M18 12h4" />
      <path d="M4.93 19.07l2.83-2.83" />
      <path d="M16.24 7.76l2.83-2.83" />
    </svg>
  ),
  layers: (s) => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={s}>
      <polygon points="12 2 2 7 12 12 22 7 12 2" />
      <polyline points="2 17 12 22 22 17" />
      <polyline points="2 12 12 17 22 12" />
    </svg>
  ),
  code: (s) => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={s}>
      <polyline points="16 18 22 12 16 6" />
      <polyline points="8 6 2 12 8 18" />
    </svg>
  ),
  chart: (s) => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={s}>
      <line x1="18" y1="20" x2="18" y2="10" />
      <line x1="12" y1="20" x2="12" y2="4" />
      <line x1="6" y1="20" x2="6" y2="14" />
      <line x1="2" y1="20" x2="22" y2="20" />
    </svg>
  ),
  terminal: (s) => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={s}>
      <polyline points="4 17 10 11 4 5" />
      <line x1="12" y1="19" x2="20" y2="19" />
    </svg>
  ),
  doc: (s) => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={s}>
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
      <line x1="16" y1="13" x2="8" y2="13" />
      <line x1="16" y1="17" x2="8" y2="17" />
      <polyline points="10 9 9 9 8 9" />
    </svg>
  ),
  package: (s) => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={s}>
      <line x1="16.5" y1="9.4" x2="7.5" y2="4.21" />
      <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
      <polyline points="3.27 6.96 12 12.01 20.73 6.96" />
      <line x1="12" y1="22.08" x2="12" y2="12" />
    </svg>
  ),
};

const CONTACT_ICON = { "@": "mail", BI: "briefcase", AS: "calendar", SZ: "pin", WEB: "globe" };
const RESEARCH_ICON = { "3D": "cube", AI: "brain", PAR: "sun", UAV: "satellite", DT: "layers" };
const SKILL_ICON = { PY: "code", "3D": "cube", RS: "satellite", AI: "brain", SIM: "chart", DEV: "terminal" };
const OUTPUT_ICON = { PP: "doc", SW: "package" };

function MarkIcon({ name, size = 18 }) {
  const fn = ICONS[name];
  return fn ? fn({ width: size, height: size }) : null;
}

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
  const iconName = CONTACT_ICON[mark];

  const content = (
    <>
      <span className={styles.contactMark} aria-hidden="true">
        {iconName ? <MarkIcon name={iconName} size={16} /> : mark}
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
  const iconName = SKILL_ICON[group.mark];
  return (
    <article className={styles.skillCard}>
      <div className={styles.skillHeader}>
        <span className={styles.skillMark} aria-hidden="true">
          {iconName ? <MarkIcon name={iconName} size={14} /> : group.mark}
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

  const [visibleSection, setVisibleSection] = React.useState("research-focus");
  const contentRef = React.useRef(null);

  React.useEffect(() => {
    const container = contentRef.current;
    if (!container) return undefined;

    // Collect all section IDs from navigation items
    const sectionIds = copy.navigation.items.map((item) =>
      item.href.replace("#", "")
    );

    // Find the actual DOM elements that have these IDs (may be on headings inside sections)
    const targets = sectionIds
      .map((id) => container.querySelector(`[id="${id}"]`))
      .filter(Boolean);

    if (!targets.length) return undefined;

    const observer = new IntersectionObserver(
      (entries) => {
        const intersecting = entries
          .filter((entry) => entry.isIntersecting)
          .sort(
            (a, b) => a.boundingClientRect.top - b.boundingClientRect.top
          );
        if (!intersecting.length) return;
        const target = intersecting[0].target;
        setVisibleSection((prev) =>
          prev === target.id ? prev : target.id
        );
      },
      {
        rootMargin: "-30% 0px -60% 0px",
        threshold: [0, 0.25, 0.5],
      }
    );

    targets.forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, [copy.navigation.items]);

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
      mark: "AS",
      label: copy.contacts.assistant,
      hint: copy.contacts.assistantHint,
      value: cvIdentity.assistantEmail,
      href: `mailto:${cvIdentity.assistantEmail}`,
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

        <div ref={contentRef} className={styles.shell}>
          <header className={styles.hero}>
            <div className={styles.heroIdentity}>
              <div className={styles.avatarFrame}>
                <img
                  src="/img/cv_person.png?v=2"
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
            {copy.navigation.items.map((item, index) => {
              const sectionId = item.href.replace("#", "");
              const isActive = visibleSection === sectionId;
              return (
                <Link
                  key={item.href}
                  to={item.href}
                  className={isActive ? styles.navLinkActive : ""}
                  aria-current={isActive ? "location" : undefined}
                >
                  <span aria-hidden="true">
                    {String(index + 1).padStart(2, "0")}
                  </span>
                  {item.label}
                </Link>
              );
            })}
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
                        {RESEARCH_ICON[item.mark] ? (
                          <MarkIcon name={RESEARCH_ICON[item.mark]} size={16} />
                        ) : (
                          item.mark
                        )}
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

            <section
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
            </section>
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
                      {OUTPUT_ICON[output.mark] ? (
                        <MarkIcon name={OUTPUT_ICON[output.mark]} size={20} />
                      ) : (
                        output.mark
                      )}
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
