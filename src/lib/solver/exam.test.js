/**
 * Parsing and comparison of exam answers (exam.js).
 *
 * Run: npm run test:science
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  answerLabel,
  examPrompt,
  matchesKey,
  normalizeAnswer,
  parseExamAnswer,
  sameAnswer,
} from "./exam.js";

test("JSON answers are parsed, also inside code fences", () => {
  const reply =
    '```json\n{"question":"稀释浓硫酸…","question_type":"single","answer":"B","reason":"酸入水","basis":"","confidence":"high"}\n```';
  const a = parseExamAnswer(reply);
  assert.equal(a.type, "single");
  assert.deepEqual(a.answer, { kind: "letters", letters: "B" });
  assert.equal(a.confidence, "high");
  assert.equal(a.reason, "酸入水");
});

test("multiple-choice letters are normalized and order-free", () => {
  for (const raw of ["ACD", "A, C, D", "D、A、C", ["C", "A", "D"], "a c d"]) {
    assert.deepEqual(normalizeAnswer(raw, "multiple"), {
      kind: "letters",
      letters: "ACD",
    });
  }
});

test("true/false in English and Chinese", () => {
  for (const raw of [true, "true", "正确", "对", "√"]) {
    assert.deepEqual(normalizeAnswer(raw, "true_false"), {
      kind: "bool",
      value: true,
    });
  }
  for (const raw of [false, "False", "错误", "错", "×"]) {
    assert.deepEqual(normalizeAnswer(raw, "true_false"), {
      kind: "bool",
      value: false,
    });
  }
  // F stays an option letter, not "false".
  assert.deepEqual(normalizeAnswer("F", "single"), {
    kind: "letters",
    letters: "F",
  });
});

test("plain-text replies fall back to the answer line", () => {
  const a = parseExamAnswer("分析……\n答案：B\n理由：……");
  assert.deepEqual(a.answer, { kind: "letters", letters: "B" });
  assert.equal(parseExamAnswer("I cannot read the image."), null);
});

test("two answers are compared after normalization", () => {
  assert.ok(sameAnswer(normalizeAnswer("CA"), normalizeAnswer("A、C")));
  assert.ok(!sameAnswer(normalizeAnswer("AC"), normalizeAnswer("ACD")));
  assert.ok(sameAnswer(normalizeAnswer("正确"), normalizeAnswer(true)));
  assert.ok(!sameAnswer(normalizeAnswer("B"), normalizeAnswer(true)));
  assert.ok(matchesKey(normalizeAnswer("错误"), false));
  assert.equal(answerLabel(normalizeAnswer(true), { yes: "正确" }), "正确");
});

test("the check prompt differs from the main prompt", () => {
  const main = examPrompt({ text: "Q" });
  const check = examPrompt({ text: "Q", variant: "check" });
  assert.notEqual(main, check);
  assert.match(check, /each option/);
  assert.match(examPrompt({}), /in the image/);
});
