import React, {Component} from "react";
import {Button, Divider, Grid, Label, Segment} from "semantic-ui-react";
import {cropText} from "../Util";
import DeleteAgentModal from "./DeleteAgentModal";

export default class AgentShortDetail extends Component {
  // props:
  // me: User
  // author: User
  // agent: Agent

  isContributor = () => {
    if (this.props.me === undefined) return false;
    if (this.props.me.id === this.props.agent.author) return false;
    let isContributor = false;
    this.props.agent.contributors.forEach(contributor => {
      if (contributor.user === this.props.me.id) {
        isContributor = true;
      }
    });
    return isContributor;
  };

  getLabels = () => {
    let labels = [];
    if (!this.props.agent.public) {
      labels.push(<Label color='yellow'>Private</Label>);
    }
    if (this.props.agent.public) {
      labels.push(<Label color='green'>Public</Label>);
    }
    if (this.props.me !== undefined && this.props.me.id === this.props.author.id) {
      labels.push(<Label>You are the author</Label>);
    }
    if (this.isContributor()) {
      labels.push(<Label>You are a contributor</Label>);
    }

    return (<span>{labels}</span>);
  };

  render = () => (
    <Segment textAlign='left'>
      <span>
        <Grid>
          <Grid.Column width={10}>
            <h3>
              <a href={'/agent/' + this.props.agent.id}>
              {this.props.author.username + ' / ' + this.props.agent.name}
              </a>
            </h3>
          </Grid.Column>
          <Grid.Column textAlign='right' width={6}>
          {this.getLabels()}
          </Grid.Column>
        </Grid>
      </span>
      <span>{cropText(this.props.agent.description, 128)}</span>
      <br/>
      <Divider/>
      <Grid columns={2}>
        <Grid.Column>
          <Button
            primary
            download
            href={this.props.agent.parameters}
            icon='cloud download'
            content='Download'
            size='mini'
          />
          <Button
            color='orange'
            href={'/agent/' + this.props.agent.id}
            icon='bars'
            content='Details'
            size='mini'
          />
        </Grid.Column>
        <Grid.Column>
          <div style={{textAlign: 'right'}}>
            <DeleteAgentModal me={this.props.me} small agent={this.props.agent}/>
          </div>
        </Grid.Column>
      </Grid>
    </Segment>
  );
}
