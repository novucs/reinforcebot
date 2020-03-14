import React, {Component} from "react";
import {createAgent} from "../Util";
import {toast} from "react-semantic-toasts";
import {Button, Form, Header, Icon, List, Modal, Popup} from "semantic-ui-react";

export default class CreateAgentModal extends Component {
  constructor(props) {
    // props:
    // onCreate()
    super(props);
    this.state = this.initialState();
    this.parametersFileInput = React.createRef();
  }

  initialState = () => {
    return {
      creating: false,
      name: '',
      description: '',
      parametersFile: null,
    };
  };

  resetState = () => {
    this.setState(this.initialState);
  };

  submit = () => {
    createAgent(this.state.name, this.state.description, this.state.parametersFile, () => {
      this.resetState();

      toast(
        {
          type: 'success',
          title: 'Success',
          description: <p>Agent successfully created</p>
        },
      );

      if (this.props.onCreate !== undefined) {
        this.props.onCreate();
      }
    });
  };

  onFileChange = () => {
    let file = this.parametersFileInput.current.files[0];
    if (!file.name.endsWith('.tar.gz')) {
      toast(
        {
          type: 'error',
          title: 'Bad file',
          description: <p>You must select a valid agent file</p>
        },
      );
      return;
    }
    this.setState({parametersFile: file});
  };

  unableToCreateAgentReasons = () => {
    let reasons = [];
    if (this.state.name === '') {
      reasons.push((<List.Item key='1'>The agent must have a name</List.Item>));
    }

    if (this.state.description === '') {
      reasons.push((<List.Item key='2'>The agent must have a description</List.Item>));
    }

    if (this.state.parametersFile === null) {
      reasons.push((<List.Item key='3'>Agent training parameters must be provided</List.Item>));
    }

    return reasons;
  };

  render = () => {
    return (
      <Modal open={this.state.creating}
             trigger={
               <Button icon positive onClick={() =>
                 this.setState({creating: true})
               }>
                 <Icon name='plus'/>{} Create a new agent
               </Button>
             }
             basic
             size='small'>
        <Header icon='add square' content='Create Agent'/>
        <Modal.Content>
          <p>Create an agent to backup on our cloud service.</p>
          <Form.Input
            fluid
            required
            icon="tag"
            iconPosition="left"
            placeholder="Name"
            onChange={event => this.setState({name: event.target.value})}
          />
          <p/>
          <Form.Input
            fluid
            required
            icon="pencil"
            iconPosition="left"
            placeholder="Description"
            onChange={event => this.setState({description: event.target.value})}
          />
          <p/>
          <Popup
            flowing
            position='right center'
            trigger={
              <Button
                content={this.state.parametersFile === null ? "Choose File" : this.state.parametersFile.name}
                labelPosition="left"
                icon="file"
                color='green'
                onClick={() => this.parametersFileInput.current.click()}
              />
            }>
            Find your agent parameter files
            under: <br/><code>/home/&lt;username&gt;/.agents/&lt;name&gt;-&lt;timestamp&gt;.tar.gz</code>
          </Popup>
          <input
            ref={this.parametersFileInput}
            type="file"
            hidden
            onChange={this.onFileChange}
          />
        </Modal.Content>
        <Modal.Actions>
          <Button basic color='grey' inverted onClick={() => this.resetState()}>
            <Icon name='remove'/> Cancel
          </Button>
          <Popup
            flowing
            position='bottom right'
            disabled={this.unableToCreateAgentReasons().length === 0}
            trigger={
              <span>
              <Button
                color='green'
                inverted
                disabled={this.unableToCreateAgentReasons().length !== 0}
                onClick={() => this.submit()}
              >
                <Icon name='checkmark'/> Create
              </Button>
            </span>
            }>
            <List bulleted>
              {this.unableToCreateAgentReasons()}
            </List>
          </Popup>
        </Modal.Actions>
      </Modal>
    );
  };
}
