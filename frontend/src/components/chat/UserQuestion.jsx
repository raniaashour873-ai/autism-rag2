function UserQuestion({ question }) {
  return (
    <article className="message user-message">

      <div className="message-label">
        YOUR QUESTION
      </div>

      <div className="user-question-content">
        {question}
      </div>

    </article>
  );
}

export default UserQuestion;