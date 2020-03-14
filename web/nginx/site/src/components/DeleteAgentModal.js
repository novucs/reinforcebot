import React, {Component} from "react";
import {Button, Header, Icon, Modal} from "semantic-ui-react";
import {deleteAgent} from "../Util";

export default class DeleteAgentModal extends Component {
  constructor(props) {
    // props:
    // agent: Agent
    // small: boolean
    super(props);
    this.state = {deleting: false};
  }

  trigger() {
    if (this.props.small) {
      return (
        <Button icon onClick={() => {
          this.setState({deleting: true})
        }} color='red'>
          <Icon name='cancel'/>
        </Button>
      );
    }

    return (
      <Button
        fluid
        style={{marginTop: '5px'}}
        color='red'
        icon='cancel'
        content='Delete'
        onClick={() => this.setState({deleting: true})}
      />
    );
  }

  render = () => {
    return (
      <Modal open={this.state.deleting}
             trigger={this.trigger()}
             basic
             size='small'>
        <Header icon='cancel' content='Delete Agent'/>
        <Modal.Content>
          <b>Warning:</b> Deleting the agent <b>"{this.props.agent.name}"</b>, are you sure you want to do this?
        </Modal.Content>
        <Modal.Actions>
          <Button basic color='grey' inverted onClick={() => this.setState({deleting: false})}>
            Cancel
          </Button>
          <Button
            color='red'
            inverted
            onClick={() => deleteAgent(this.props.agent.id, () => window.location = '/dashboard')}>
            Delete
          </Button>
        </Modal.Actions>
      </Modal>
    );
  }
}
