from typing import Literal
from pydantic import BaseModel, Field, model_validator

Mode = Literal['explain', 'plan', 'quiz']

class Plan(BaseModel):
    query: str = Field(min_length=1, max_length=500)

    @model_validator(mode='after')
    def nonblank_query(self):
        if not self.query.strip():
            raise ValueError('query must not be blank')
        return self

class Citation(BaseModel):
    source_id: int


    version: str = ''
    section: str = ''

class ChoiceQuestion(BaseModel):
    text: str = Field(min_length=1)
    options: list[str] = Field(min_length=2, max_length=5)
    correct: int = Field(ge=0, strict=True)

    @model_validator(mode='after')
    def valid_index(self):
        if self.correct >= len(self.options):
            raise ValueError('correct is outside options')
        if not self.text.strip():
            raise ValueError('question fields must not be blank')
        normalized = [option.strip().casefold() for option in self.options]
        if any(not option for option in normalized) or len(set(normalized)) != len(normalized):
            raise ValueError('options must be nonempty and distinct')
        return self

class Question(ChoiceQuestion):
    explanation: str = Field(min_length=1)
    topic: str = Field(min_length=1)

    @model_validator(mode='after')
    def nonempty_details(self):
        if not self.explanation.strip() or not self.topic.strip():
            raise ValueError('question details must not be blank')
        return self

class ProposedQuestion(ChoiceQuestion):
    evidence_id: int = Field(ge=0)

class QuizProposal(BaseModel):
    questions: list[ProposedQuestion] = Field(min_length=3, max_length=3)

    @model_validator(mode='after')
    def distinct_questions(self):
        if len({q.text.strip().casefold() for q in self.questions}) != len(self.questions):
            raise ValueError('quiz questions must be distinct')
        return self

class Draft(BaseModel):
    text: str = Field(min_length=1)
    citations: list[Citation] = Field(min_length=1)
    questions: list[Question] = Field(default_factory=list, max_length=5)

    @model_validator(mode='after')
    def nonblank_text(self):
        if not self.text.strip():
            raise ValueError('draft text must not be blank')
        return self

class QuizDraft(Draft):
    questions: list[Question] = Field(min_length=3, max_length=3)

    @model_validator(mode='after')
    def distinct_questions(self):
        if len({q.text.strip().casefold() for q in self.questions}) != len(self.questions):
            raise ValueError('quiz questions must be distinct')
        return self

class ClaimCheck(BaseModel):
    statement_id: int = Field(ge=0)
    source_id: int
    quote: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    supported: bool

    @model_validator(mode='after')
    def nonblank_evidence(self):
        if not self.quote.strip() or not self.reason.strip():
            raise ValueError('evidence and reason must not be blank')
        return self

class Verdict(BaseModel):

    checks: list[ClaimCheck]
    issues: list[str] = Field(default_factory=list)
    approved: bool

    @model_validator(mode='after')
    def rejection_has_reason(self):
        if any(not issue.strip() for issue in self.issues):
            raise ValueError('issues must contain meaningful reasons')
        if not self.approved and not self.issues:
            raise ValueError('rejection needs issues')
        if self.approved and self.issues:
            raise ValueError('approval cannot contain issues')
        return self
