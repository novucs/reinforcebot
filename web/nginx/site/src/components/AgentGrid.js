import React, {Component} from "react";
import AgentShortDetail from "./AgentShortDetail";

export default class AgentGrid extends Component {
  // props:
  // me: User
  // users: Object[str, User]
  // agents: List[Agent]

  render() {
    let components = [];
    this.props.agents.forEach(agent => {
      if (!(agent.author in this.props.users)) {
        return;
      }
      let author = this.props.users[agent.author];
      components.push((
        <AgentShortDetail me={this.props.me} agent={agent} author={author}/>
      ));
    });
    return (
      <p style={{marginTop: '16px', marginBottom: '16px'}}>
        {components}
      </p>
    );
  };
}
