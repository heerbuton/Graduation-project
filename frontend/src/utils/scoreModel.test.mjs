import test from "node:test";
import assert from "node:assert/strict";

import { convertLlmResultToScoreModel } from "./scoreModel.js";


test("convertLlmResultToScoreModel should split measures and map guqin fields", () => {
  const llmResult = [
    {
      new_measure: true,
      pitch: "1",
      octave: "4",
      duration: "4",
      action: "",
      string: "一",
      position: "一",
      finger: "",
    },
    {
      pitch: "6",
      octave: "3",
      duration: "8",
      action: "历",
      string: "六",
      position: "",
      finger: "",
    },
    {
      new_measure: true,
      pitch: "5",
      octave: "3",
      duration: "4",
      action: "",
      string: "五",
      position: "五",
      finger: "",
    },
  ];

  const scoreModel = convertLlmResultToScoreModel(llmResult);
  assert.equal(scoreModel.measures.length, 2);
  assert.equal(scoreModel.measures[0].notes.length, 2);
  assert.equal(scoreModel.measures[1].notes.length, 1);
  assert.equal(scoreModel.measures[0].notes[1].guqin.stringOrder, "六");
});


test("convertLlmResultToScoreModel should fallback invalid enums in non strict mode", () => {
  const scoreModel = convertLlmResultToScoreModel(
    [{ pitch: "9", octave: "8", duration: "32" }],
    { strict: false }
  );

  const note = scoreModel.measures[0].notes[0];
  assert.equal(note.pitch, "1");
  assert.equal(note.octave, "4");
  assert.equal(note.duration, "4");
  assert.ok(scoreModel.issues.length >= 1);
});


test("convertLlmResultToScoreModel should throw in strict mode on invalid duration", () => {
  assert.throws(
    () => convertLlmResultToScoreModel([{ pitch: "1", octave: "4", duration: "32" }], { strict: true }),
    /duration/
  );
});


test("convertLlmResultToScoreModel should infer measures by duration when new_measure is missing", () => {
  const scoreModel = convertLlmResultToScoreModel([
    { pitch: "1", octave: "4", duration: "4" },
    { pitch: "2", octave: "4", duration: "4" },
    { pitch: "3", octave: "4", duration: "4" },
    { pitch: "4", octave: "4", duration: "4" },
    { pitch: "5", octave: "4", duration: "4" },
  ]);

  assert.equal(scoreModel.measureCount, 2);
  assert.equal(scoreModel.measures[0].notes.length, 4);
  assert.equal(scoreModel.measures[1].notes.length, 1);
});


test("convertLlmResultToScoreModel should still infer when only first note has new_measure", () => {
  const scoreModel = convertLlmResultToScoreModel([
    { new_measure: true, pitch: "1", octave: "4", duration: "4" },
    { pitch: "2", octave: "4", duration: "4" },
    { pitch: "3", octave: "4", duration: "4" },
    { pitch: "4", octave: "4", duration: "4" },
    { pitch: "5", octave: "4", duration: "4" },
  ]);

  assert.equal(scoreModel.measureCount, 2);
  assert.equal(scoreModel.measures[0].notes.length, 4);
  assert.equal(scoreModel.measures[1].notes.length, 1);
});
