import React, { useEffect, useId, useRef, useState } from "react";
import Link from "@docusaurus/Link";
import {
  AI_CATALOG_CHECKED_AT,
  AI_PROVIDER_GROUPS,
  AI_PROVIDER_PRESETS,
  changeAIProvider,
  getAIProvider,
  localized,
} from "../../lib/api";
import styles from "./styles.module.css";
import { makeToolText } from "@site/src/lib/i18n/toolText";
import ZH from "./zh";

const tx = makeToolText(ZH);

const OTHER_MODEL = "__other__";

const PROTOCOL_LABELS = {
  openai: "OpenAI Chat Completions",
  "openai-responses": "OpenAI Responses",
  anthropic: "Anthropic Messages",
  gemini: "Gemini generateContent",
};

const GROUPED_PROVIDERS = AI_PROVIDER_GROUPS.map((group) => ({
  ...group,
  providers: AI_PROVIDER_PRESETS.filter((item) => item.group === group.id),
})).filter((group) => group.providers.length > 0);

export default function AIProviderSettings({
  value,
  onChange,
  title = "AI model",
  step,
  requireVision = false,
}) {
  const provider = getAIProvider(value.provider);
  const instanceId = useId();
  const titleId = `${instanceId}-title`;
  const isDemo = provider.protocol === "mock";
  const latestState = useRef({ onChange, value });
  const selectedModel = provider.models.find(
    (model) => model.id === value.model
  );
  const [otherChosen, setOtherChosen] = useState(false);
  const showModelInput =
    provider.models.length === 0 || otherChosen || !selectedModel;
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

  const changeProvider = (providerId) => {
    setOtherChosen(false);
    onChange(changeAIProvider(value, providerId));
  };

  const chooseModel = (modelId) => {
    if (modelId === OTHER_MODEL) {
      setOtherChosen(true);
      updateField("model", "");
      return;
    }
    setOtherChosen(false);
    updateField("model", modelId);
  };

  const modelOptionLabel = (model) =>
    `${model.label} · ${model.vision ? tx("image + text") : tx("text only")}`;

  return (
    <section className={styles.panel} aria-labelledby={titleId}>
      <div className={styles.headingRow}>
        <div>
          <span className={styles.eyebrow}>
            {step || tx("Bring your own key")}
          </span>
          <h3 id={titleId} className={styles.title}>
            {title}
          </h3>
        </div>
        <span className={styles.status}>
          {isDemo ? tx("Private demo") : tx("Experimental API")}
        </span>
      </div>

      <div className={styles.grid}>
        <label className={styles.field}>
          <span>{tx("Provider")}</span>
          <select
            value={provider.id}
            onChange={(event) => changeProvider(event.target.value)}
          >
            {GROUPED_PROVIDERS.map((group) => (
              <optgroup key={group.id} label={localized(group.label)}>
                {group.providers.map((item) => (
                  <option key={item.id} value={item.id}>
                    {localized(item.name)}
                  </option>
                ))}
              </optgroup>
            ))}
          </select>
        </label>

        {!isDemo && provider.models.length > 0 && (
          <label className={styles.field}>
            <span>{tx("Model")}</span>
            <select
              value={showModelInput ? OTHER_MODEL : value.model}
              onChange={(event) => chooseModel(event.target.value)}
              aria-invalid={visionMismatch || undefined}
            >
              {provider.models.map((model) => (
                <option key={model.id} value={model.id}>
                  {modelOptionLabel(model)}
                </option>
              ))}
              <option value={OTHER_MODEL}>{tx("Other model ID…")}</option>
            </select>
          </label>
        )}

        {!isDemo && showModelInput && (
          <label
            className={
              provider.models.length > 0
                ? `${styles.field} ${styles.fullWidth}`
                : styles.field
            }
          >
            <span>{tx("Model ID")}</span>
            <input
              value={value.model}
              onChange={(event) => updateField("model", event.target.value)}
              placeholder={
                provider.group === "local"
                  ? tx("e.g. the model tag you pulled or loaded")
                  : tx("Enter the provider's model ID")
              }
              autoComplete="off"
              spellCheck="false"
            />
          </label>
        )}

        {!isDemo && (
          <label className={`${styles.field} ${styles.fullWidth}`}>
            <span>{tx("API endpoint")}</span>
            <input
              type="url"
              value={
                provider.endpointLocked ? provider.endpoint : value.endpoint
              }
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
            <span>
              {provider.keyOptional ? tx("API key (optional)") : tx("API key")}
            </span>
            <span className={styles.secretRow}>
              <input
                type="password"
                value={value.apiKey}
                onChange={(event) => updateField("apiKey", event.target.value)}
                placeholder={tx("Paste the key for this tab")}
                autoComplete="off"
                spellCheck="false"
              />
              <button
                type="button"
                className={styles.clearButton}
                onClick={() => updateField("apiKey", "")}
                disabled={!value.apiKey}
              >
                {tx("Clear")}
              </button>
            </span>
          </label>
        )}
      </div>

      {!isDemo && (
        <ul className={styles.facts}>
          <li>
            <span>{tx("API format")}</span>
            {PROTOCOL_LABELS[provider.protocol]}
          </li>
          {selectedModel && (
            <li>
              <span>{tx("Input")}</span>
              {selectedModel.vision ? tx("image + text") : tx("text only")}
            </li>
          )}
          {provider.docs && (
            <li>
              <Link to={provider.docs}>{tx("Models and pricing ↗")}</Link>
            </li>
          )}
        </ul>
      )}

      <p className={styles.description}>{localized(provider.description)}</p>
      {visionMismatch && (
        <p className={styles.warning} role="alert">
          {tx(
            "This model does not accept image input. Choose a model marked “image + text” or use text mode."
          )}
        </p>
      )}
      {requireVision && !selectedModel && !isDemo && (
        <p className={styles.warning}>
          {tx(
            "Image support for this custom model ID is unknown; verify it with the provider."
          )}
        </p>
      )}
      <p className={styles.privacyNote}>
        {isDemo
          ? tx("No network request is made in demo mode.")
          : tx(
              "Experimental BYOK: the key is kept only in this tab’s memory, cleared on refresh or exit, and sent only to the endpoint shown above. A static website cannot protect it like a backend can. Use a restricted test key; for production, use your own authenticated proxy."
            )}{" "}
        {tx(
          "Provider presets follow the OpenClaw and models.dev catalogs (checked {0}); any other model the provider serves can be entered by ID.",
          AI_CATALOG_CHECKED_AT
        )}
      </p>
    </section>
  );
}
