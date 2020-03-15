import React, {Component} from "react";
import {Button, Divider, Grid, Label, Segment} from "semantic-ui-react";
import {cropText} from "../Util";
import DeleteAgentModal from "./DeleteAgentModal";

export default class AgentShortDetail extends Component {
  // props:
  // me: User
  // author: User
  // agent: Agent

  render = () => (
    <Segment textAlign='left'>
      <Label color='green' ribbon>
        Created by {this.props.author.username} ({this.props.author.first_name} {this.props.author.last_name})
      </Label>
      <br/>
      <span><h1>{this.props.author.username + ' / ' + this.props.agent.name}</h1></span>
      <br/>
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
            size='medium'
          />
          <Button
            color='orange'
            href={'/agent/' + this.props.agent.id}
            icon='bars'
            content='Details'
            size='medium'
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
