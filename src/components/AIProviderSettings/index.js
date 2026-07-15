import React, { useEffect, useId, useRef } from "react";
import {
  AI_PROVIDER_PRESETS,
  changeAIProvider,
  getAIProvider,
} from "../../lib/api";
import styles from "./styles.module.css";

export default function AIProviderSettings({
  value,
  onChange,
  title = "AI model",
  requireVision = false,
}) {
  const provider = getAIProvider(value.provider);
  const instanceId = useId();
  const titleId = `${instanceId}-title`;
  const listId = `${instanceId}-models`;
  const isDemo = provider.protocol === "mock";
  const latestState = useRef({ onChange, value });
  const selectedModel = provider.models.find(
    (model) => model.id === value.model
  );
  const visionMismatch =
    requireVision && selectedModel && !selectedModel.vision;

  useEffect(() => {
    latestState.current = { onChange, value };
  }, [onChange, value]);

  useEffect(() => {
    const clearForPageExit = () => {
      const current = latestState.current;
      if (current.value.apiKey) {
        current.onChange({ ...current.value, apiKey: "" });
      }
    };
    window.addEventListener("pagehide", clearForPageExit);
    return () => window.removeEventListener("pagehide", clearForPageExit);
  }, []);

  const updateField = (field, nextValue) => {
    onChange({ ...value, [field]: nextValue });
  };

  return (
    <section className={styles.panel} aria-labelledby={titleId}>
      <div className={styles.headingRow}>
        <div>
          <span className={styles.eyebrow}>Bring your own key</span>
          <h3 id={titleId} className={styles.title}>
            {title}
          </h3>
        </div>
        <span className={styles.status}>
          {isDemo ? "Private demo" : "Experimental API"}
        </span>
      </div>

      <div className={styles.grid}>
        <label className={styles.field}>
          <span>Provider</span>
          <select
            value={provider.id}
            onChange={(event) =>
              onChange(changeAIProvider(value, event.target.value))
            }
          >
            {AI_PROVIDER_PRESETS.map((item) => (
              <option key={item.id} value={item.id}>
                {item.name}
              </option>
            ))}
          </select>
        </label>

        {!isDemo && (
          <label className={styles.field}>
            <span>Model</span>
            <input
              value={value.model}
              onChange={(event) => updateField("model", event.target.value)}
              list={listId}
              placeholder="Enter a model ID"
              autoComplete="off"
              spellCheck="false"
              aria-invalid={visionMismatch || undefined}
            />
            <datalist id={listId}>
              {provider.models.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.label}
                </option>
              ))}
            </datalist>
          </label>
        )}

        {!isDemo && (
          <label className={`${styles.field} ${styles.fullWidth}`}>
            <span>API endpoint</span>
            <input
              type="url"
              value={value.endpoint}
              onChange={(event) => updateField("endpoint", event.target.value)}
              placeholder="https://your-provider.example/v1/chat/completions"
              autoComplete="url"
              spellCheck="false"
              readOnly={provider.endpointLocked}
            />
          </label>
        )}

        {!isDemo && (
          <label className={`${styles.field} ${styles.fullWidth}`}>
            <span>API key</span>
            <span className={styles.secretRow}>
              <input
                type="password"
                value={value.apiKey}
                onChange={(event) => updateField("apiKey", event.target.value)}
                placeholder="Paste the key for this tab"
                autoComplete="off"
                spellCheck="false"
              />
              <button
                type="button"
                className={styles.clearButton}
                onClick={() => updateField("apiKey", "")}
                disabled={!value.apiKey}
              >
                Clear
              </button>
            </span>
          </label>
        )}
      </div>

      <p className={styles.description}>{provider.description}</p>
      {visionMismatch && (
        <p className={styles.warning} role="alert">
          This preset is not marked as vision-capable. Choose a vision model or
          use text mode.
        </p>
      )}
      {requireVision && !selectedModel && !isDemo && (
        <p className={styles.warning}>
          Image support for this custom model ID is unknown; verify it with the
          provider.
        </p>
      )}
      <p className={styles.privacyNote}>
        {isDemo
          ? "No network request is made in demo mode."
          : "Experimental BYOK: the key is kept only in this tab\u2019s memory, cleared on refresh or exit, and sent only to the endpoint shown above. A static website cannot protect it like a backend can. Use a restricted test key; for production, use your own authenticated proxy."}
      </p>
    </section>
  );
}
